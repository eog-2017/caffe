#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_triplet_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include "boost/thread.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "boost/fusion/include/map.hpp"
#include "boost/fusion/include/set.hpp"

namespace caffe {

template<typename Dtype>
ImageTripletDataLayer<Dtype>::~ImageTripletDataLayer<Dtype>() {
	this->StopInternalThread();
}

template<typename Dtype>
void ImageTripletDataLayer<Dtype>::DataLayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int new_height = this->layer_param_.image_data_param().new_height();
	const int new_width = this->layer_param_.image_data_param().new_width();
	const bool is_color = this->layer_param_.image_data_param().is_color();
	string root_folder = this->layer_param_.image_data_param().root_folder();

	TransformationParameter transform_param =
			this->layer_param_.transform_param();
	CHECK((new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0))
																									<< "Current implementation requires new_height and new_width to be set at the same time.";

	string image_set_folder = root_folder + "ImageSets/Main/";

	boost::regex pattern("(.*)_train.txt");
	boost::smatch what;

	/**
	 * Stores the locations for each object which can be fetched
	 * randomly while we prepare the batches.
	 */
	for (boost::filesystem::recursive_directory_iterator iter(image_set_folder),
			end; iter != end; ++iter) {

		std::string name = iter->path().filename().string();
		if (regex_match(name, what, pattern)) {
			LOG(INFO)<< what[1];
			std::ifstream infile(iter->path().c_str());
			string index;
			string presence;
			list list;
			while (infile >> index >> presence) if (presence == "1") list.list.push_back(index);

			objects_[what[1].str()] = list;
		}

	}

	// Read the file with filenames and labels
	CHECK(this->layer_param_.image_data_param().has_source());
	const string& source = image_set_folder
			+ this->layer_param_.image_data_param().source();
	LOG(INFO)<< "Opening file " << source;
	std::ifstream infile(source.c_str());
	string filename;
	string label;
	while (infile >> filename >> label) {
		lines_.push_back(std::make_pair(filename, label));
	}

	if (this->layer_param_.image_data_param().shuffle()) {
		// randomly shuffle data
		LOG(INFO)<< "Shuffling data";
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		ShuffleImages();
	}

	LOG(INFO)<< "A total of " << lines_.size() << " images.";

	lines_id_ = 0;
	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.image_data_param().rand_skip()) {
		unsigned int skip = caffe_rng_rand()
				% this->layer_param_.image_data_param().rand_skip();
		LOG(INFO)<< "Skipping first " << skip << " data points.";
		CHECK_GT(lines_.size(), skip)<< "Not enough points to skip";
		lines_id_ = skip;
	}

	CHECK_EQ(is_color, true);
	const int channels = 3;

	const int batch_size = this->layer_param_.image_data_param().batch_size();
	CHECK_GT(batch_size, 0)<< "Positive batch size required";

	top[0]->Reshape(batch_size, channels, new_height, new_width);
	this->transformed_data_.Reshape(batch_size, channels, new_height, new_width);

	LOG(INFO)<< "output anchor size: " << top[0]->num() << ","
	<< top[0]->channels() << "," << top[0]->height() << ","
	<< top[0]->width();

	top[1]->Reshape(batch_size, channels, new_height, new_width);
	this->transformed_positive_.Reshape(batch_size, channels, new_height, new_width);
	LOG(INFO)<< "output positive size: " << top[1]->num() << ","
	<< top[1]->channels() << "," << top[1]->height() << ","
	<< top[1]->width();

	top[2]->Reshape(batch_size, channels, new_height, new_width);
	this->transformed_negative_.Reshape(batch_size, channels, new_height, new_width);
	LOG(INFO)<< "output negative size: " << top[2]->num() << ","
	<< top[2]->channels() << "," << top[2]->height() << ","
	<< top[2]->width();

}

template <typename Dtype>
void ImageTripletDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      this->prefetch_[i]->label_.mutable_cpu_data();
      this->prefetch_[i]->label__.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        this->prefetch_[i]->label_.mutable_gpu_data();
        this->prefetch_[i]->label__.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  this->StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void ImageTripletDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!this->must_stop()) {
      Batch<Dtype>* batch = this->prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
          batch->label__.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      this->prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void ImageTripletDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->prefetch_current_) {
    this->prefetch_free_.push(this->prefetch_current_);
  }
  this->prefetch_current_ = this->prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(this->prefetch_current_->data_);
  top[0]->set_cpu_data(this->prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(this->prefetch_current_->label_);
    top[1]->set_cpu_data(this->prefetch_current_->label_.mutable_cpu_data());
    top[2]->ReshapeLike(this->prefetch_current_->label__);
    top[2]->set_cpu_data(this->prefetch_current_->label__.mutable_cpu_data());
  }
}

template<typename Dtype>
void ImageTripletDataLayer<Dtype>::ShuffleImages() {
	caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());
	for (map<string, list>::iterator it = objects_.begin();
			it != objects_.end(); ++it) {
		shuffle(it->second.list.begin(), it->second.list.end(), prefetch_rng);
		it->second.it = it->second.list.begin();
	}
}

// This function is called on prefetch thread
template<typename Dtype>
void ImageTripletDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());
	CHECK(this->transformed_data_.count());

	ImageDataParameter image_data_param = this->layer_param_.image_data_param();
	const int batch_size = image_data_param.batch_size();
	const int new_height = image_data_param.new_height();
	const int new_width = image_data_param.new_width();
	const bool is_color = image_data_param.is_color();

	string root_folder = image_data_param.root_folder();

	CHECK_EQ(is_color, true);
	const int channels = 3;

	batch->data_.Reshape(batch_size, channels, new_height, new_width);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();
	Dtype* prefetch_positive = batch->label_.mutable_cpu_data();
	Dtype* prefetch_negative = batch->label__.mutable_cpu_data();

	map<string, list>::iterator it = objects_.begin();

	/**
	 * Serially fetch an item from the objects_ map
	 */
	int item_id = 0;
	int count = 0;
	int lines_size = 4000; // TODO:REMOVE THIS LATER
	while (item_id < batch_size) {

		if(count > objects_.size()) {
			ShuffleImages();
		}

		if (objects_[it->first].it == objects_[it->first].list.end()) {
			it++;
			count++;
			continue;
		}

		count = 0;

		if ((objects_[it->first].it)+1 == objects_[it->first].list.end()) {
			string anchor = *(objects_[it->first].it);
			(objects_[it->first].it)--;
			string positive = *(objects_[it->first].it);
			objects_[it->first].it = objects_[it->first].list.end();
		} else {
			string anchor = *(objects_[it->first].it);
			(objects_[it->first].it)++;
			string positive = *(objects_[it->first].it);
		}

		map<string, list>::iterator t_it;
		do {
			t_it = objects_.begin();
			std::advance(t_it, rand() % objects_.size());
		} while (t_it != it);

		vector<string>::iterator n_it = objects_[t_it->first].list.begin();
		std::advance(n_it, rand() % objects_.size());
		string negative = *n_it;

		it++;
		item_id++;

		std::vector<cv::Mat> cv_img_seg;
		// get a blob
		timer.Start();
		CHECK_GT(lines_size, lines_id_);

		//int img_row, img_col; //Use these to store the actual size of image for later resizing
		cv_img_seg.push_back(
				ReadImageToCVMat(root_folder + lines_[lines_id_].first,
						new_height, new_width, is_color));

		if (!cv_img_seg[0].data) {
			DLOG(INFO)<< "Fail to load img: " << root_folder + lines_[lines_id_].first;
		}

		cv_img_seg.push_back(
				ReadImageToCVMat(root_folder + lines_[lines_id_].second,
						false));
		if (!cv_img_seg[1].data) {
			DLOG(INFO)<< "Fail to load seg: " << root_folder + lines_[lines_id_].second;
		}

		read_time += timer.MicroSeconds();
		timer.Start();

		// Apply transformations (mirror, crop...) to the image
		int offset = batch->data_.offset(item_id);
		this->transformed_data_.set_cpu_data(prefetch_data + offset);

		offset = batch->label_.offset(item_id);
		this->transformed_positive_.set_cpu_data(prefetch_positive + offset);

		trans_time += timer.MicroSeconds();

		// go to the next iter
		lines_id_++;
		if (lines_id_ >= lines_size) {
			// We have reached the end. Restart from the first.
			DLOG(INFO)<< "Restarting data prefetching from start.";
			lines_id_ = 0;
		}

	}

	batch_timer.Stop();
	DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageTripletDataLayer);
REGISTER_LAYER_CLASS(ImageTripletData);

}  // namespace caffe
#endif  // USE_OPENCV
