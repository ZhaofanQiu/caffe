
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/14
** desc： Volume data layer
*********************************************************************************/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	VolumeDataLayer<Dtype>::VolumeDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param){
	}

	template <typename Dtype>
	VolumeDataLayer<Dtype>::~VolumeDataLayer() {
		this->StopInternalThread();
	}

	template <typename Dtype>
	void VolumeDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Initialize the leveldb
		leveldb::DB* db_temp;
		leveldb::Options options;
		options.create_if_missing = false;
		options.max_open_files = 100;
		LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
		leveldb::Status status = leveldb::DB::Open(
			options, this->layer_param_.data_param().source(), &db_temp);
		CHECK(status.ok()) << "Failed to open leveldb "
			<< this->layer_param_.data_param().source() << std::endl
			<< status.ToString();
		db_.reset(db_temp);
		iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
		iter_->SeekToFirst();
		// Check if we would need to randomly skip a few data points
		if (this->layer_param_.data_param().rand_skip()) {
			unsigned int skip = caffe_rng_rand() %
				this->layer_param_.data_param().rand_skip();
			LOG(INFO) << "Skipping first " << skip << " data points.";
			while (skip-- > 0) {
				iter_->Next();
				if (!iter_->Valid()) {
					iter_->SeekToFirst();
				}
			}
		}
		// Read a data point, and use it to initialize the top blob.
		VolumeDatum datum;
		datum.ParseFromString(iter_->value().ToString());
		// image
		const int batch_size = this->layer_param_.video_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		int crop_size = this->layer_param_.data_param().crop_size();
		const unsigned int prefetch_rng_seed = caffe_rng_rand();
		prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
		if (crop_size > 0) {
			top_shape_ = video_shape(this->layer_param_.data_param().batch_size(),
				datum.channels(), datum.length(), crop_size, crop_size);
		}
		else {
			top_shape_ = video_shape(this->layer_param_.data_param().batch_size(),
				datum.channels(), datum.length(), datum.height(), datum.width());
		}
		LOG(INFO) << "output data size: " << top_shape_[0] << ","
			<< top_shape_[1] << "," << top_shape_[2] << "," << top_shape_[3] << ","
			<< top_shape_[4];
		// label
		vector<int> label_shape = vector<int>(1, this->layer_param_.data_param().batch_size());
		int datum_channels_ = datum.channels();
		int datum_length_ = datum.length();
		int datum_height_ = datum.height();
		int datum_width_ = datum.width();
		int datum_size_ = datum.channels() * datum.length() * datum.height() * datum.width();
		this->origin_height_ = datum_height_;
		this->origin_width_ = datum_width_;

		// datum size
		CHECK_GT(top_shape_[3], crop_size);
		CHECK_GT(top_shape_[4], crop_size);
		// check if we want to have mean
		if (this->layer_param_.data_param().has_mean_file()) {
			const string& mean_file = this->layer_param_.data_param().mean_file();
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
		}

		top[0]->Reshape(top_shape_);
		for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
			this->prefetch_[i].data_.Reshape(top_shape_);
		}
		// label
		if (this->output_labels_) {
			vector<int> label_shape(1, batch_size);
			top[1]->Reshape(label_shape);
			for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
				this->prefetch_[i].label_.Reshape(label_shape);
			}
		}
	}

	template <typename Dtype>
	unsigned int VolumeDataLayer<Dtype>::PrefetchRand() {
		CHECK(prefetch_rng_);
		caffe::rng_t* prefetch_rng =
			static_cast<caffe::rng_t*>(prefetch_rng_->generator());
		return (*prefetch_rng)();
	}

	// This function is called on prefetch thread
	template<typename Dtype>
	void VolumeDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
		CPUTimer batch_timer;
		batch_timer.Start();
		double read_time = 0;
		double trans_time = 0;
		CPUTimer timer;
		CHECK(batch->data_.count());

		VolumeDatum datum;

		const Dtype scale = this->layer_param_.data_param().scale();
		const int batch_size = this->layer_param_.data_param().batch_size();
		const int crop_size = this->layer_param_.data_param().crop_size();
		const bool mirror = this->layer_param_.data_param().mirror();

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
		Dtype* prefetch_label = batch->label_.mutable_cpu_data();

		const Dtype* mean = this->data_mean_.cpu_data();
		char *data_buffer = NULL;
		for (int item_id = 0; item_id < batch_size; ++item_id) {
			timer.Start();
			// get a blob
			CHECK(this->iter_);
			CHECK(this->iter_->Valid());
			datum.ParseFromString(this->iter_->value().ToString());

			read_time += timer.MicroSeconds();
			timer.Start();
			int offset = batch->data_.offset(vector<int>(1, item_id));
			Dtype* top_data = prefetch_data + offset;
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
					}
				}
				else {
					for (int j = 0; j < size; ++j) {
						top_data[item_id * size + j] =
							(datum.float_data(j) - mean[j]) * scale;
					}
				}
			}

			if (this->output_labels_) {
				prefetch_label[item_id] = datum.label();
			}
			// go to the next iteration
			this->iter_->Next();
			if (!this->iter_->Valid()) {
				// We have reached the end. Restart from the first.
				DLOG(INFO) << "Restarting data prefetching from start.";
				this->iter_->SeekToFirst();
			}
			trans_time += timer.MicroSeconds();
		}
		timer.Stop();
		batch_timer.Stop();
		DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
		DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
		DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
	}

	INSTANTIATE_CLASS(VolumeDataLayer);
	REGISTER_LAYER_CLASS(VolumeData);
}  // namespace caffe
