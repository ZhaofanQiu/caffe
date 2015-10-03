
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/10/3
** desc： Test lstm speed 
*********************************************************************************/

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{

	template <typename Dtype>
	class GridLSTMLayerTest {
	public:
		GridLSTMLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~GridLSTMLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(8, 16, 8, 16, 32));
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;

	public:
		void StartTest(int iters){
			LayerParameter layer_param;
			InnerProductParameter* ip_param = layer_param.mutable_inner_product_param();
			DropoutParameter* dp_param = layer_param.mutable_dropout_param();
			GridLSTMParameter* grid_param = layer_param.mutable_grid_lstm_param();
			ip_param->set_num_output(4);
			FillerParameter* fp1 = ip_param->mutable_weight_filler();
			fp1->set_type("gaussian");
			fp1->set_std(0.01);
			FillerParameter* fp2 = ip_param->mutable_bias_filler();
			fp2->set_type("constant");
			fp2->set_value(0);
			dp_param->set_dropout_ratio(0.5);
			grid_param->set_num_output(4);
			grid_param->add_reverse(true);
			grid_param->add_reverse(true);
			grid_param->add_reverse(true);


			shared_ptr<Layer<Dtype>> layer(new GridLSTMLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			for (int i = 0; i < iters; i++)
			{
				layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
				layer->Backward(this->blob_top_vec_, vector<bool>(1, true), this->blob_bottom_vec_);
			}
			EXPECT_EQ(this->blob_top_->shape(0), 8);
			EXPECT_EQ(this->blob_top_->shape(1), 4);
			EXPECT_EQ(this->blob_top_->shape(2), 8);
			EXPECT_EQ(this->blob_top_->shape(3), 16);
			EXPECT_EQ(this->blob_top_->shape(4), 32);
		}
	};
}

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::GridLSTMLayerTest<float> test;
	caffe::CPUTimer timer;
	timer.Start();
	test.StartTest(100);
	timer.Stop();
	LOG(INFO) << "LSTM time: " << timer.MilliSeconds() << " ms.";
	return 0;
}
