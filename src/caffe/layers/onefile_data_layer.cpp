
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu 
** mail： zhaofanqiu@gmail.com 
** date： 2015/12/1
** desc： Onefiledata layer 
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
	OnefileDataLayer<Dtype>::~OnefileDataLayer<Dtype>()
	{
		this->StopInternalThread();
	}

	template <typename Dtype>
	void OnefileDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// Read the file with filenames and labels
		source_ = this->layer_param_.video_data_param().source();
		LOG(INFO) << "Opening file " << source_;
		// Read a data point, and use it to initialize the top blob.
		const int batch_size = this->layer_param_.video_data_param().batch_size();
		CHECK_GT(batch_size, 0) << "Positive batch size required";
		this->fp_ = fopen(source_.c_str(), "rb");
		int n_dim;
		fread(&n_dim, sizeof(int), 1, fp_);
		top_shape_ = vector<int>(n_dim, 0);
		fread(&top_shape_[0], sizeof(int), n_dim, fp_);
		top_shape_[0] = batch_size;

		// Reshape prefetch_data and top[0] according to the batch_size
		for (int i = 0; i < this->PREFETCH_COUNT; ++i)
		{
			this->prefetch_[i].data_.Reshape(top_shape_);
		}
		top[0]->Reshape(top_shape_);
		_fseeki64(fp_, 0, SEEK_SET);
	}

	// This function is called on prefetch thread
	template <typename Dtype>
	void OnefileDataLayer<Dtype>::load_batch(Batch<Dtype>* batch)
	{
		CHECK(batch->data_.count());
		batch->data_.Reshape(top_shape_);
		// datum scales
		int size = batch->data_.count(1);

		Dtype* prefetch_data = batch->data_.mutable_cpu_data();

		for (int item_id = 0; item_id < top_shape_[0]; ++item_id) {
			int n_dim;
			if (!fread(&n_dim, sizeof(int), 1, fp_))
			{
				_fseeki64(fp_, 0, SEEK_SET);
				CHECK(fread(&n_dim, sizeof(int), 1, fp_));
			}
			vector<int> dims(n_dim, 0);
			CHECK_EQ(n_dim, fread(&dims[0], sizeof(int), n_dim, fp_));

			int offset = batch->data_.offset(vector<int>(1, item_id));
			Dtype* top_data = prefetch_data + offset;

			CHECK_EQ(size, fread(top_data, sizeof(Dtype), size, fp_));
			if (feof(fp_))
			{
				_fseeki64(fp_, 0, SEEK_SET);
			}
		}
	}

	INSTANTIATE_CLASS(OnefileDataLayer);
	REGISTER_LAYER_CLASS(OnefileData);

}  // namespace caffe
