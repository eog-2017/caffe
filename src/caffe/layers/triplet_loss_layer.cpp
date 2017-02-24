#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK(this->layer_param_.triplet_loss_param().has_alpha());
	alpha = this->layer_param().triplet_loss_param().alpha();
}

template<typename Dtype>
void TripletLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))<< "Inputs must have the same dimension.";
	CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))<< "Inputs must have the same dimension.";

	a_n_.ReshapeLike(*bottom[0]);
	p_n_.ReshapeLike(*bottom[0]);
	n_n_.ReshapeLike(*bottom[0]);

	p_diff_.ReshapeLike(*bottom[0]);
	n_diff_.ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();

	/**
	 * Normalize the vectors
	 */
	Dtype len_a = caffe_cpu_dot(count, a_n_.cpu_data(), a_n_.cpu_data());
	a_n_.scale_data(1 / len_a);
	Dtype len_p = caffe_cpu_dot(count, p_n_.cpu_data(), p_n_.cpu_data());
	p_n_.scale_data(1 / len_p);
	Dtype len_n = caffe_cpu_dot(count, n_n_.cpu_data(), n_n_.cpu_data());
	n_n_.scale_data(1 / len_n);

	caffe_sub(count, a_n_.cpu_data(), p_n_.cpu_data(),
			p_diff_.mutable_cpu_data());
	caffe_sub(count, a_n_.cpu_data(), n_n_.cpu_data(),
			n_diff_.mutable_cpu_data());
	Dtype positive = caffe_cpu_dot(count, p_diff_.cpu_data(),
			p_diff_.cpu_data());
	Dtype negative = caffe_cpu_dot(count, n_diff_.cpu_data(),
			n_diff_.cpu_data());

	Dtype loss = positive - negative + alpha / bottom[0]->num() / Dtype(2);
	top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	for (int i = 0; i < 3; ++i) {
		if (propagate_down[i]) {
			const Dtype sign = (i == 0) ? 1 : -1;
			const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			caffe_cpu_axpby(bottom[i]->count(),              // count
					alpha,                              // alpha
					p_diff_.cpu_data(),                   // a
					Dtype(0),                           // beta
					bottom[i]->mutable_cpu_diff());  // b
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
