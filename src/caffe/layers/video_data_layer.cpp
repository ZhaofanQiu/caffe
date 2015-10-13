
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

	vector<int> video_shape(int num, int channels, int length, int height, int width)
	{
		vector<int> shape(5, 0);
		shape[0] = num;
		shape[1] = channels;
		shape[2] = length;
		shape[3] = height;
		shape[4] = width;
		return shape;
	}

	template <typename Dtype>
	VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>()
	{
		this->StopInternalThread();
	}

	template <typename Dtype>
	void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int new_length = this->layer_param_.video_data_param().new_length();
		const int new_height = this->layer_param_.video_data_param().new_height();
		const int new_width = this->layer_param_.video_data_param().new_width();
		const int sampling_rate = this->layer_param_.video_data_param().sampling_rate();

		CHECK(new_length > 0) << "new length need to be positive";
		CHECK((new_height == 0 && new_width == 0) ||
			(new_height > 0 && new_width > 0)) << "Current implementation requires "
			"new_height and new_width to be set at the same time.";

		// Read the file with filenames and labels
		const string& source = this->layer_param_.video_data_param().source();
		const bool use_temporal_jitter = this->layer_param_.video_data_param().use_temporal_jitter();
		const bool use_image = this->layer_param_.video_data_param().use_image();
		LOG(INFO) << "Opening file " << source;
		std::ifstream infile(source.c_str());
		int count = 0;
		string filename;
		int start_frm, label;

		if ((!use_image) && use_temporal_jitter){
			while (infile >> filename >> label) {
				file_list_.push_back(filename);
				label_list_.push_back(label);
				shuffle_index_.push_back(count);
				count++;
			}
		}
		else {
			while (infile >> filename >> start_frm >> label) {
				file_list_.push_back(filename);
				start_frm_list_.push_back(start_frm);
				label_list_.push_back(label);
				shuffle_index_.push_back(count);
				count++;
			}
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
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.video_data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.video_data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			CHECK_GT(shuffle_index_.size(), skip) << "Not enough points to skip";
			lines_id_ = skip;
		}
		// Read a data point, and use it to initialize the top blob.
		VolumeDatum datum;
		int id = shuffle_index_[lines_id_];
		if (!use_image){
			if (use_temporal_jitter){
				srand(time(NULL));
				CHECK(ReadVideoToVolumeDatum(file_list_[0].c_str(), 0, label_list_[0],
					new_length, new_height, new_width, sampling_rate, &datum));
			}
			else
				CHECK(ReadVideoToVolumeDatum(file_list_[id].c_str(), start_frm_list_[id], label_list_[id],
				new_length, new_height, new_width, sampling_rate, &datum));
		}
		else{

			LOG(INFO) << "read video from " << file_list_[id].c_str();
			CHECK(ReadImageSequenceToVolumeDatum(file_list_[id].c_str(), start_frm_list_[id], label_list_[id],
				new_length, new_height, new_width, sampling_rate, &datum));
		}

		// image
		int crop_size = this->layer_param_.video_data_param().crop_size();
		const int batch_size = this->layer_param_.video_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		top_shape_ = vector<int>(5, 0);
		top_shape_[0] = batch_size;
		top_shape_[1] = datum.channels();
		top_shape_[2] = datum.length();
		if (crop_size > 0) {
			top_shape_[3] = crop_size;
			top_shape_[4] = crop_size;
		}
		else {
			top_shape_[3] = datum.height();
			top_shape_[4] = datum.width();
		}
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

		// datum size
		int datum_channels_ = datum.channels();
		int datum_length_ = datum.length();
		int datum_height_ = datum.height();
		int datum_width_ = datum.width();
		int datum_size_ = datum.channels() * datum.length() * datum.height() * datum.width();
		this->origin_height_ = datum_height_;
		this->origin_width_ = datum_width_;
		CHECK_GT(datum_height_, crop_size);
		CHECK_GT(datum_width_, crop_size);
		// check if we want to have mean
		if (this->layer_param_.video_data_param().has_mean_file()) {
			const string& mean_file = this->layer_param_.video_data_param().mean_file();
			LOG(INFO) << "Loading mean file from" << mean_file;
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
			data_mean_.FromProto(blob_proto);
			CHECK_EQ(data_mean_.shape(0), 1);
			CHECK_EQ(data_mean_.shape(1), datum_channels_);
			CHECK_EQ(data_mean_.shape(2), datum_length_);
			CHECK_EQ(data_mean_.shape(3), datum_height_);
			CHECK_EQ(data_mean_.shape(4), datum_width_);
		}
		else {
			// Simply initialize an all-empty mean.
			data_mean_.Reshape(video_shape(1, datum_channels_, datum_length_, datum_height_, datum_width_));
			if (this->layer_param_.video_data_param().has_mean_value()){
				LOG(INFO) << "Using mean value of " << this->layer_param_.video_data_param().mean_value();
				caffe::caffe_set(data_mean_.count(), (Dtype)this->layer_param_.video_data_param().mean_value(),
					(Dtype*)data_mean_.mutable_cpu_data());
			}
		}
	}

	template <typename Dtype>
	void VideoDataLayer<Dtype>::ShuffleVideo()
	{
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		shuffle(this->shuffle_index_.begin(), this->shuffle_index_.end(), prefetch_rng);
	}

	template <typename Dtype>
	unsigned int VideoDataLayer<Dtype>::PrefetchRand() {
		CHECK(prefetch_rng_);
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		return (*prefetch_rng)();
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
	{
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());

		VideoDataParameter video_data_param = this->layer_param_.video_data_param();
		const Dtype scale = video_data_param.scale();
		const int batch_size = video_data_param.batch_size();
		const int crop_size = video_data_param.crop_size();
		const bool mirror = video_data_param.mirror();
		const int new_length = video_data_param.new_length();
		const int new_height = video_data_param.new_height();
		const int new_width = video_data_param.new_width();
		const bool use_image = video_data_param.use_image();
		const int sampling_rate = video_data_param.sampling_rate();
		const bool use_temporal_jitter = video_data_param.use_temporal_jitter();
		if (mirror && crop_size == 0) {
			LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
				<< "set at the same time.";
		}
		
		batch->data_.Reshape(top_shape_);
		// datum scales
		const int channels = top_shape_[1];
		const int length = top_shape_[2];
		const int height = this->origin_height_;
		const int width = this->origin_width_;
		int size = channels * length * height * width;

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();

		VolumeDatum datum;
		const int chunks_size = this->shuffle_index_.size();
		const Dtype* mean = this->data_mean_.cpu_data();
		const int show_data = video_data_param.show_data();
		char *data_buffer = NULL;
		if (show_data)
			data_buffer = new char[size];
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			// get a blob
			timer.Start();
			CHECK_GT(chunks_size, this->lines_id_);
			bool read_status;
			int id = this->shuffle_index_[this->lines_id_];
			if (!use_image){
				if (!use_temporal_jitter){
					read_status = ReadVideoToVolumeDatum(this->file_list_[id].c_str(), this->start_frm_list_[id],
						this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
				}
				else{
					read_status = ReadVideoToVolumeDatum(this->file_list_[id].c_str(), -1,
						this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
				}
			}
			else {
				if (!use_temporal_jitter) {
					read_status = ReadImageSequenceToVolumeDatum(this->file_list_[id].c_str(), this->start_frm_list_[id],
						this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
				}
				else {
					int num_of_frames = this->start_frm_list_[id];
					int use_start_frame;
					if (num_of_frames<new_length*sampling_rate){
						LOG(INFO) << "not enough frames; having " << num_of_frames;
						read_status = false;
					}
					else {
						if (this->phase_ == Phase::TRAIN)
							use_start_frame = this->PrefetchRand() % (num_of_frames - new_length*sampling_rate + 1) + 1;
						else
							use_start_frame = 0;

						read_status = ReadImageSequenceToVolumeDatum(this->file_list_[id].c_str(), use_start_frame,
							this->label_list_[id], new_length, new_height, new_width, sampling_rate, &datum);
					}
				}
			}

			if (this->phase_ == Phase::TEST){
				CHECK(read_status) << "Testing must not miss any example";
			}

			if (!read_status) {
				//LOG(ERROR) << "cannot read " << this->file_list_[id];
				this->lines_id_++;
				if (this->lines_id_ >= chunks_size) {
					// We have reached the end. Restart from the first.
					DLOG(INFO) << "Restarting data prefetching from start.";
					this->lines_id_ = 0;
					if (video_data_param.shuffle()){
						this->ShuffleVideo();
					}
				}
				item_id--;
				continue;
			}
			read_time += timer.MicroSeconds();

			timer.Start();
			int offset = batch->data_.offset(vector<int>(1, item_id));
			Dtype* top_data = prefetch_data + offset;
			//LOG(INFO) << "--> " << item_id;
			//LOG(INFO) << "label " << datum.label();
			const string& data = datum.data();
			if (crop_size) {
				CHECK(data.size()) << "Image cropping only support uint8 data";
				int h_off, w_off;
				// We only do random crop when we do training.
				if (this->phase_ == Phase::TRAIN) {
					h_off = this->PrefetchRand() % (height - crop_size);
					w_off = this->PrefetchRand() % (width - crop_size);
				}
				else {
					h_off = (height - crop_size) / 2;
					w_off = (width - crop_size) / 2;
				}
				if (mirror && this->PrefetchRand() % 2) {
					// Copy mirrored version
					for (int c = 0; c < channels; ++c) {
						for (int l = 0; l < length; ++l) {
							for (int h = 0; h < crop_size; ++h) {
								for (int w = 0; w < crop_size; ++w) {
									int top_index = (((item_id * channels + c) * length + l) * crop_size + h)
										* crop_size + (crop_size - 1 - w);
									int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
									Dtype datum_element =
										static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
									top_data[top_index] = (datum_element - mean[data_index]) * scale;
									if (show_data)
										data_buffer[((c * length + l) * crop_size + h)
										* crop_size + (crop_size - 1 - w)] = static_cast<uint8_t>(data[data_index]);
								}
							}
						}
					}
				}
				else {
					// Normal copy
					for (int c = 0; c < channels; ++c) {
						for (int l = 0; l < length; ++l) {
							for (int h = 0; h < crop_size; ++h) {
								for (int w = 0; w < crop_size; ++w) {
									int top_index = (((item_id * channels + c) * length + l) * crop_size + h)
										* crop_size + w;
									int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
									Dtype datum_element =
										static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
									top_data[top_index] = (datum_element - mean[data_index]) * scale;
									if (show_data)
										data_buffer[((c * length + l) * crop_size + h)
										* crop_size + w] = static_cast<uint8_t>(data[data_index]);
								}
							}
						}
					}
				}
			}
			else {
				// we will prefer to use data() first, and then try float_data()
				if (data.size()) {
					for (int j = 0; j < size; ++j) {
						Dtype datum_element =
							static_cast<Dtype>(static_cast<uint8_t>(data[j]));
						top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
						if (show_data)
							data_buffer[j] = static_cast<uint8_t>(data[j]);
					}
				}
				else {
					for (int j = 0; j < size; ++j) {
						top_data[item_id * size + j] =
							(datum.float_data(j) - mean[j]) * scale;
					}
				}
			}
			trans_time += timer.MicroSeconds();

			if (show_data>0){
				int image_size, channel_size;
				if (crop_size){
					image_size = crop_size * crop_size;
				}
				else{
					image_size = height * width;
				}
				channel_size = length * image_size;
				for (int l = 0; l < length; ++l) {
					for (int c = 0; c < channels; ++c) {
						cv::Mat img;
						char ch_name[64];
						if (crop_size)
							BufferToGrayImage(data_buffer + c * channel_size + l * image_size, crop_size, crop_size, &img);
						else
							BufferToGrayImage(data_buffer + c * channel_size + l * image_size, height, width, &img);
						sprintf(ch_name, "Channel %d", c);
						cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
						cv::imshow(ch_name, img);
					}
					cv::waitKey(100);
				}
			}
			if (this->output_labels_) {
				Dtype* prefetch_label = batch->label_.mutable_cpu_data();
				prefetch_label[item_id] = datum.label();
				 //LOG(INFO) << "fetching label" << datum.label() << std::endl;
			}

			this->lines_id_++;
			if (this->lines_id_ >= chunks_size) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				this->lines_id_ = 0;
				if (video_data_param.shuffle()){
					this->ShuffleVideo();
				}
			}
		}
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
		if (show_data & data_buffer != NULL)
			delete[]data_buffer;
	}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
