
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu 
** mail： zhaofanqiu@gmail.com 
** date： 2015/9/3 
** desc： Videodata layer 
*********************************************************************************/

#include <stdint.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/video_layers.hpp"

using std::string;

namespace caffe {

	template <typename Dtype>
	BinDataLayer<Dtype>::~BinDataLayer<Dtype>()
	{
		this->StopInternalThread();
	}

	template <typename Dtype>
	void BinDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		length_ = this->layer_param_.video_data_param().new_length();
		height_ = this->layer_param_.video_data_param().new_height();
		width_ = this->layer_param_.video_data_param().new_width();

		CHECK(length_ > 0) << "new length need to be positive";
		CHECK(height_ > 0) << "new height need to be positive";
		CHECK(width_ > 0) << "new width need to be positive";

		// Read the file with filenames and labels
		const string& source = this->layer_param_.video_data_param().source();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		int count = 0;
		string filename;
		int label;

		while (infile >> filename >> label) {
			file_list_.push_back(filename);
			label_list_.push_back(label);
			shuffle_index_.push_back(count);
			count++;
		}

		if (count == 0){
			LOG(INFO) << "failed to read chunk list" << std::endl;
		}

		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		if (this->layer_param_.video_data_param().shuffle()){
			LOG(INFO) << "Shuffling data";
			ShuffleVideo();
		}
		LOG(INFO) << "A total of " << shuffle_index_.size() << " video chunks.";

		lines_id_ = 0;
		// Read a data point, and use it to initialize the top blob.
		const int batch_size = this->layer_param_.video_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";

		this->channels_ = 0;
		FILE* fp = fopen(file_list_[0].c_str(), "rb");
		vector<Dtype> buff(length_ * height_ * width_, 0);
		while (!feof(fp))
		{
			const int size_buff = fread(&buff[0], sizeof(Dtype), length_ * height_ * width_, fp);
			if (size_buff != length_ * height_ * width_)
			{
				break;
			}
			this->channels_++;
		}
		fclose(fp);
		top_shape_ = vector<int>(5, 0);
		top_shape_[0] = batch_size;
		top_shape_[1] = this->channels_;
		top_shape_[2] = this->length_;
		top_shape_[3] = this->height_;
		top_shape_[4] = this->width_;
		// Reshape prefetch_data and top[0] according to the batch_size
		for (int i = 0; i < this->PREFETCH_COUNT; ++i)
		{
			this->prefetch_[i].data_.Reshape(top_shape_);
		}
		top[0]->Reshape(top_shape_);

		LOG(INFO) << "output data size: " << top[0]->shape(0) << ","
			<< top[0]->shape(1) << "," << top[0]->shape(2) << "," << top[0]->shape(3) << ","
			<< top[0]->shape(4);

		// label
		if (this->output_labels_) {
			vector<int> label_shape(1, batch_size);
			top[1]->Reshape(label_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i)
			{
				this->prefetch_[i].label_.Reshape(label_shape);
			}
		}
	}

	template <typename Dtype>
	void BinDataLayer<Dtype>::ShuffleVideo()
	{
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(this->shuffle_index_.begin(), this->shuffle_index_.end(), prefetch_rng);
	}

	template <typename Dtype>
	unsigned int BinDataLayer<Dtype>::PrefetchRand() {
		CHECK(prefetch_rng_);
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		return (*prefetch_rng)();
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void BinDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
	{
		CHECK(batch->data_.count());
		VideoDataParameter video_data_param = this->layer_param_.video_data_param();
		const Dtype scale = video_data_param.scale();
		const int batch_size = video_data_param.batch_size();
		
		batch->data_.Reshape(top_shape_);
		// datum scales
		int size = channels_ * length_ * height_ * width_;

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();

		const int chunks_size = this->shuffle_index_.size();

		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			CHECK_GT(chunks_size, this->lines_id_);
			int id = this->shuffle_index_[this->lines_id_];

			int offset = batch->data_.offset(vector<int>(1, item_id));
			Dtype* top_data = prefetch_data + offset;

			FILE* fp = fopen(this->file_list_[id].c_str(), "rb");
			fread(top_data, sizeof(Dtype), size, fp);
			fclose(fp);

			this->lines_id_++;
			if (this->lines_id_ >= chunks_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				this->lines_id_ = 0;
				if (video_data_param.shuffle()){
					this->ShuffleVideo();
				}
			}
			if (this->output_labels_) {
				Dtype* prefetch_label = batch->label_.mutable_cpu_data();
				prefetch_label[item_id] = this->label_list_[id];
				//LOG(INFO) << "fetching label" << datum.label() << std::endl;
			}
		}
	}

INSTANTIATE_CLASS(BinDataLayer);
REGISTER_LAYER_CLASS(BinData);

}  // namespace caffe
