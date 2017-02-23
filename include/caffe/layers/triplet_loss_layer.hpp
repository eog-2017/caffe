#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Triplet (L2) loss @f$
 *          T = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| a_n - p_n
 *			\right| \right|_2^2 + \alpha - \left| \left| a_n - n_n
 *        	\right| \right|_2^2 @f$ for real-valued regression tasks.
 *        Defined in http://arxiv.org/abs/1503.03832v1
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the targets @f$ y \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed Triplet loss: @f$ T = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| a_n - p_n
 *			\right| \right|_2^2 + \alpha - \left| \left| a_n - n_n
 *        	\right| \right|_2^2  @f$
 *
 *
 */
template<typename Dtype>
class TripletLossLayer: public LossLayer<Dtype> {
public:
	explicit TripletLossLayer(const LayerParameter& param) :
			LossLayer<Dtype>(param), a_n_(), p_n_(), n_n_(), p_diff_(), n_diff_() {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const {
		return "TripletLoss";
	}

	/**
	 * In this layer, the modification is that it needs to have 3 feature
	 * input and a Dtype output. the function ExactNumTopBlobs() need not
	 * be changed.
	 */
	virtual inline int ExactNumBottomBlobs() const {
		return 3;
	}

	/**
	 * Unlike most loss layers, in the TripletLossLayer we must backpropagate
	 * to all inputs -- override to return true and always allow force_backward.
	 */
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

protected:
	/// @copydoc TripletLossLayer
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	/**
	 * @brief
	 *
	 * @param top output Blob vector (length 1), providing the error gradient with
	 *      respect to the outputs
	 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
	 *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
	 *      as @f$ \lambda @f$ is the coefficient of this layer's output
	 *      @f$\ell_i@f$ in the overall Net loss
	 *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
	 *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
	 *      (*Assuming that this top Blob is not used as a bottom (input) by any
	 *      other layer of the Net.)
	 * @param propagate_down see Layer::Backward.
	 * @param bottom input Blob vector (length 2)
	 *   -# @f$ (N \times C \times H \times W) @f$
	 *      the predictions @f$\hat{y}@f$; Backward fills their diff with
	 *      gradients @f$
	 *        \frac{\partial E}{\partial \hat{y}} =
	 *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
	 *      @f$ if propagate_down[0]
	 *   -# @f$ (N \times C \times H \times W) @f$
	 *      the targets @f$y@f$; Backward fills their diff with gradients
	 *      @f$ \frac{\partial E}{\partial y} =
	 *          \frac{1}{n} \sum\limits_{n=1}^N (y_n - \hat{y}_n)
	 *      @f$ if propagate_down[1]
	 */
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down,
			const vector<Blob<Dtype>*>& bottom);

	Dtype alpha;

	/**
	 * || a_n || [|| a_n || == 1]
	 * || p_n || [|| p_n || == 1]
	 * || n_n || [|| n_n || == 1]
	 */
	Blob<Dtype> a_n_;
	Blob<Dtype> p_n_;
	Blob<Dtype> n_n_;

	/**
	 * || a_n - p_n ||2 has to be minimized
	 * and
	 * || a_n - n_n ||2 has to be maximized
	 */
	Blob<Dtype> p_diff_;
	Blob<Dtype> n_diff_;
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
