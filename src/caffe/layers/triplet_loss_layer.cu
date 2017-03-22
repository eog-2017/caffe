#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  caffe_copy(count, bottom[0]->gpu_data(), a_n_.mutable_gpu_data());
  caffe_copy(count, bottom[1]->gpu_data(), p_n_.mutable_gpu_data());
  caffe_copy(count, bottom[2]->gpu_data(), n_n_.mutable_gpu_data());

  /**
   * Normalize the vectors
   */
  Dtype len_a;
  caffe_gpu_dot(count, a_n_.gpu_data(), a_n_.gpu_data(), &len_a);
  a_n_.scale_data(1 / len_a);
  Dtype len_p;
  caffe_gpu_dot(count, p_n_.gpu_data(), p_n_.gpu_data(), &len_p);
  p_n_.scale_data(1 / len_p);
  Dtype len_n;
  caffe_gpu_dot(count, n_n_.gpu_data(), n_n_.gpu_data(), &len_n);
  n_n_.scale_data(1 / len_n);

  caffe_gpu_sub(count, a_n_.gpu_data(), p_n_.gpu_data(),
            p_diff_.mutable_gpu_data());
  caffe_gpu_sub(count, a_n_.gpu_data(), n_n_.gpu_data(),
            n_diff_.mutable_gpu_data());
  Dtype positive;
  caffe_gpu_dot(count, p_diff_.gpu_data(), p_diff_.gpu_data(), &positive);
  Dtype negative;
  caffe_gpu_dot(count, n_diff_.gpu_data(), n_diff_.gpu_data(), &negative);

  Dtype loss = positive - negative + alpha / bottom[0]->num() / Dtype(2);
  top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < 3; ++i) {
        bool toggle = true;
        if (propagate_down[i]) {
            Dtype sign;
            if(toggle) sign = 1;
            else sign = -1;
            toggle = !toggle;
            const Dtype alpha = sign * top[0]->gpu_diff()[0] / bottom[i]->num();
            if(i == 0)
                caffe_gpu_axpby(bottom[i]->count(),              // count
                                alpha,                              // alpha
                                p_diff_.gpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_gpu_diff());     // b
            else if(i == 1)
                caffe_gpu_axpby(bottom[i]->count(),              // count
                                alpha,                              // alpha
                                p_diff_.gpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_gpu_diff());     // b
            else
                caffe_gpu_axpby(bottom[i]->count(),              // count
                                alpha,                              // alpha
                                n_diff_.gpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_gpu_diff());     // b

        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);

}  // namespace caffe
