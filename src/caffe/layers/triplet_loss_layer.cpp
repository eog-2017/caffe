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
    LOG(INFO) << "top";
    int count = bottom[0]->count();

    caffe_copy(count, bottom[0]->cpu_data(), a_n_.mutable_cpu_data());
    caffe_copy(count, bottom[1]->cpu_data(), p_n_.mutable_cpu_data());
    caffe_copy(count, bottom[2]->cpu_data(), n_n_.mutable_cpu_data());

    LOG(INFO) << "anchor" << bottom[0]->mutable_cpu_data()[1];
    LOG(INFO) << "copy anchor" << a_n_.mutable_cpu_data()[1];

    LOG(INFO) << "count " << a_n_.shape(0);
    LOG(INFO) << "length " << a_n_.shape(1);

    /**
     * @brief length of the feature vector that is incoming
     */
    int length = a_n_.shape(1);
    int size = a_n_.shape(0);

    /**
     * Normalize the vectors in a for loop
     */

    Dtype* anchor = a_n_.mutable_cpu_data();
    Dtype* positive = p_n_.mutable_cpu_data();
    Dtype* negative = n_n_.mutable_cpu_data();

//    for (int i = 0; i < count; i++) {
//        LOG(INFO) << "a_n_["<< i <<"] :" << a_n_.cpu_data()[i];
//    }


    for (int i = 0; i < size; i++) {
        //int offset = length * i;

//        for (int k = offset; k < length + offset; k++) {
//            LOG(INFO) << "a_n_["<< k <<"] :" << a_n_.cpu_data()[k];
//        }

        Dtype len_a = caffe_cpu_dot(length, anchor, anchor);
        LOG(INFO) << "len_a  " << len_a;
        caffe_scal(length, 1.f/(Dtype)sqrt(len_a), anchor);
//        Dtype len_norm_a = caffe_cpu_dot(length, anchor, anchor);
//        DLOG(INFO) << "len_norm_a " << len_norm_a;

        Dtype len_p = caffe_cpu_dot(length, positive, positive);
        DLOG(INFO) << "len_p " << len_p;
        caffe_scal(length, 1.f/(Dtype)sqrt(len_p), positive);
//        Dtype len_norm_p = caffe_cpu_dot(length, positive, positive);
//        DLOG(INFO) << "len_norm_p " << len_norm_p;

        Dtype len_n = caffe_cpu_dot(length, negative, negative);
        DLOG(INFO) << "len_n " << len_n;
        caffe_scal(length, 1.f/(Dtype)sqrt(len_n), negative);
//        Dtype len_norm_n = caffe_cpu_dot(length, negative, negative);
//        DLOG(INFO) << "len_norm_n " << len_norm_n;

//        for (int j = offset; j < length + offset; j++) {
//            LOG(INFO) << "a_n_["<< j <<"] :" << a_n_.cpu_data()[j];
//        }

//        LOG(INFO) << "<====next iteration===>";

        anchor += length;
        positive += length;
        negative += length;

    }

    caffe_sub(count, a_n_.cpu_data(), p_n_.cpu_data(),
              p_diff_.mutable_cpu_data());
    caffe_sub(count, a_n_.cpu_data(), n_n_.cpu_data(),
              n_diff_.mutable_cpu_data());

    Dtype positive_loss = caffe_cpu_dot(count, p_diff_.cpu_data(),
                                   p_diff_.cpu_data());
    Dtype negative_loss = caffe_cpu_dot(count, n_diff_.cpu_data(),
                                   n_diff_.cpu_data());

    LOG(INFO) << "pos " << positive_loss;
    LOG(INFO) << "neg " << negative_loss;

    Dtype loss = (positive_loss - negative_loss + alpha) / bottom[0]->num();
    //Dtype loss = (positive_loss - negative_loss + alpha);
    //loss *= 0.4;
    //if (isnan(loss))
//        top[0]->mutable_cpu_data()[0] = alpha;
//    else
        top[0]->mutable_cpu_data()[0] = loss;
    LOG(INFO) << "loss : " << loss;
}

template<typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < 3; i++) {
        if (propagate_down[i]) {
            LOG(INFO) << "loss : " << top[0]->cpu_diff()[0];
            const Dtype alpha = Dtype(1) / bottom[0]->num();
            LOG(INFO) << "backward : alpha : " << alpha;
            if(i == 0)
                caffe_cpu_axpby(bottom[i]->count(),              // count
                                alpha,                              // alpha
                                p_diff_.cpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_cpu_diff());     // b
            else if(i == 1)
                caffe_cpu_axpby(bottom[i]->count(),              // count
                                -1 * alpha,                              // alpha
                                p_diff_.cpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_cpu_diff());     // b
            else
                caffe_cpu_axpby(bottom[i]->count(),              // count
                                alpha,                              // alpha
                                n_diff_.cpu_data(),                   // a
                                Dtype(0),                           // beta
                                bottom[i]->mutable_cpu_diff());     // b

        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
