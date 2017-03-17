#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

struct list {
	vector<string>::iterator it;
	vector<string> list;
};

typedef struct list list;

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageTripletDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageTripletDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageTripletDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageTripletData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  Blob<Dtype> transformed_positive_;
  Blob<Dtype> transformed_negative_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void InternalThreadEntry();
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  map<string, list> objects_;
  /**
   * Remove these two when the objects_ is working fine...
   */
  vector<std::pair<string, string> > lines_;
  int lines_id_;
};


}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
