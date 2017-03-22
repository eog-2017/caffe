#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <tinyxml.h>

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

cv::Rect getROIfromXML(string path, string object_name);

template<typename Dtype>
ImageTripletDataLayer<Dtype>::~ImageTripletDataLayer<Dtype>() {
    this->StopInternalThread();
}

template<typename Dtype>
void ImageTripletDataLayer<Dtype>::DataLayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int new_height = this->layer_param_.image_data_param().new_height();
    int new_width = this->layer_param_.image_data_param().new_width();
    const bool is_color = this->layer_param_.image_data_param().is_color();
    string root_folder = this->layer_param_.image_data_param().root_folder();

    TransformationParameter transform_param =
            this->layer_param_.transform_param();
    CHECK((new_height == 0 && new_width == 0) || (new_height > 0 && new_width > 0))
            << "Current implementation requires new_height and new_width to be set at the same time.";

    const int crop_size = transform_param.crop_size();
    new_height = crop_size;
    new_width = crop_size;

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
            DLOG(INFO)<< what[1];
            std::ifstream infile(iter->path().c_str());
            string index;
            string presence;
            list list;
            while (infile >> index >> presence) if (presence == "1") list.list.push_back(index);

            list.it = list.list.begin();

            objects_[what[1].str()] = list;
            keys_.push_back(what[1].str());
        }
        curr_key_ = keys_.begin();
    }

    if (this->layer_param_.image_data_param().shuffle()) {
        // randomly shuffle data
        LOG(INFO)<< "Shuffling data";
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleImages();
    }

    CHECK_EQ(is_color, true);
    const int channels = 3;

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    CHECK_GT(batch_size, 0)<< "Positive batch size required";

    this->transformed_data_.Reshape(batch_size, channels, new_height, new_width);

    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(batch_size, channels, new_height, new_width);
    }
    top[0]->Reshape(batch_size, channels, new_height, new_width);
    this->transformed_data_.Reshape(batch_size, channels, new_height, new_width);

    LOG(INFO)<< "output anchor size: " << top[0]->num() << ","
             << top[0]->channels() << "," << top[0]->height() << ","
             << top[0]->width();

    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(batch_size, channels, new_height, new_width);
    }
    top[1]->Reshape(batch_size, channels, new_height, new_width);
    this->transformed_positive_.Reshape(batch_size, channels, new_height, new_width);
    LOG(INFO)<< "output positive size: " << top[1]->num() << ","
             << top[1]->channels() << "," << top[1]->height() << ","
             << top[1]->width();

    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label__.Reshape(batch_size, channels, new_height, new_width);
    }
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
    shuffle(keys_.begin(), keys_.end(), prefetch_rng);
    for (map<string, list>::iterator it = objects_.begin();
         it != objects_.end(); ++it) {
        shuffle(it->second.list.begin(), it->second.list.end(), prefetch_rng);
        it->second.it = it->second.list.begin();
    }
    curr_key_ = keys_.begin();
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

    /**
     * Serially fetch an item from the objects_ map
     */
    int item_id = 0;

    while (item_id < batch_size) {

        list& curr_list = objects_[*curr_key_];

        if(curr_list.it == objects_[*curr_key_].list.end()) {
            if(++curr_key_ == keys_.end())
                ShuffleImages();
            continue;
        }

        DLOG(INFO)<<"key : "<<*curr_key_;
        string anchor = *(curr_list.it);
        DLOG(INFO)<<"anchor : "<< anchor;

        string positive;

        do{
            int random_index = rand() % curr_list.list.size();

            positive = curr_list.list[random_index];

            if (positive == anchor) continue;
            else break;

        } while(true);

        DLOG(INFO)<<"positive : "<< positive;

        string negative;
        string negative_class;

        do {
            int random_object_index = rand() % objects_.size();

            map<string, list>::iterator it = objects_.begin();
            int i = 0;
            while(i++ < random_object_index) it++;

            if (it->first == *curr_key_) continue;

            list random_list = objects_[it->first];

            int random_index = rand() % random_list.list.size();

            negative = random_list.list[random_index];
            negative_class = it->first;

            break;

        } while (true);

        DLOG(INFO)<<"negative : "<< negative;
        DLOG(INFO)<<"negative_class : "<< negative_class;

        /*
         * Read all these images from the set that you obtained and display them here.
         */
        timer.Start();
        cv::Mat cv_anchor = ReadImageToCVMat(root_folder + "JPEGImages/" + anchor + ".jpg");
        cv::Mat cv_positive = ReadImageToCVMat(root_folder + "JPEGImages/" + positive + ".jpg");
        cv::Mat cv_negative = ReadImageToCVMat(root_folder + "JPEGImages/" + negative + ".jpg");

        cv::Rect cv_rect_anchor = getROIfromXML(root_folder + "Annotations/" + anchor + ".xml", *curr_key_);
        cv::Rect cv_rect_positive = getROIfromXML(root_folder + "Annotations/" + positive + ".xml", *curr_key_);
        cv::Rect cv_rect_negative = getROIfromXML(root_folder + "Annotations/" + negative + ".xml", negative_class);

        cv::Mat cv_anchor_region = cv_anchor(cv_rect_anchor);
        cv::Mat cv_positive_region = cv_positive(cv_rect_positive);
        cv::Mat cv_negative_region = cv_negative(cv_rect_negative);

        CHECK(cv_anchor_region.data) << "Could not load " << root_folder + "JPEGImages/" + anchor + ".jpg";
        CHECK(cv_positive_region.data) << "Could not load " << root_folder + "JPEGImages/" + positive + ".jpg";
        CHECK(cv_negative_region.data) << "Could not load " << root_folder + "JPEGImages/" + negative + ".jpg";
        read_time += timer.MicroSeconds();
        timer.Start();

        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        this->data_transformer_->Transform(cv_anchor_region, &(this->transformed_data_), true);

        offset = batch->label_.offset(item_id);
        this->transformed_positive_.set_cpu_data(prefetch_positive + offset);
        this->data_transformer_->Transform(cv_positive_region, &(this->transformed_positive_), true);

        offset = batch->label__.offset(item_id);
        this->transformed_negative_.set_cpu_data(prefetch_negative + offset);
        this->data_transformer_->Transform(cv_negative_region, &(this->transformed_negative_), true);

        trans_time += timer.MicroSeconds();

        curr_list.it++;

        item_id++;
    }

    batch_timer.Stop();
    DLOG(INFO)<< "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO)<< "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO)<< "Transform time: " << trans_time / 1000 << " ms.";
}

cv::Rect getROIfromXML(string path, string object_name) {
    TiXmlDocument doc(path);

    if(!doc.LoadFile()) exit(-1);

    TiXmlElement* annotation = doc.FirstChildElement("annotation");
    TiXmlElement* object = annotation->FirstChildElement("object");
    TiXmlElement* name;
    while(object) {
        name = object->FirstChildElement("name");
        if(name->GetText() == object_name) break;
        else object = object->NextSiblingElement();
    }
    TiXmlElement* bndbox = object->FirstChildElement("bndbox");
    int xmin = std::stoi(bndbox->FirstChildElement("xmin")->GetText());
    int ymin = std::stoi(bndbox->FirstChildElement("ymin")->GetText());
    int xmax = std::stoi(bndbox->FirstChildElement("xmax")->GetText());
    int ymax = std::stoi(bndbox->FirstChildElement("ymax")->GetText());

    return cv::Rect(xmin, ymin, xmax-xmin, ymax-ymin);
}

INSTANTIATE_CLASS(ImageTripletDataLayer);
REGISTER_LAYER_CLASS(ImageTripletData);

}  // namespace caffe
#endif  // USE_OPENCV
