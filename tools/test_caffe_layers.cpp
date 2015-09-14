
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/13
** desc： Test video layers
*********************************************************************************/

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/video_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe{

	template <typename Dtype>
	class Convolution3DLayerTest {
	public:
		Convolution3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~Convolution3DLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(2, 2, 3, 10, 10));
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
		void StartTest(){
			LayerParameter layer_param;
			Convolution3DParameter* convolution_param = layer_param.mutable_convolution3d_param();
			convolution_param->set_kernel_size(4);
			convolution_param->set_kernel_l(1);
			convolution_param->set_stride(2);
			convolution_param->set_num_output(2);
			convolution_param->mutable_weight_filler()->set_type("gaussian");
			convolution_param->mutable_weight_filler()->set_std(0.01);
			convolution_param->mutable_bias_filler()->set_type("constant");
			convolution_param->mutable_bias_filler()->set_value(1.);

			shared_ptr<Layer<Dtype>> layer(new Convolution3DLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);

			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 4);
			EXPECT_EQ(this->blob_top_->shape(4), 4);
		}
	};

	template <typename Dtype>
	class Deconvolution3DLayerTest {
	public:
		Deconvolution3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~Deconvolution3DLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(2, 2, 3, 4, 4));
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
		void StartTest(){
			LayerParameter layer_param;
			Convolution3DParameter* convolution_param = layer_param.mutable_convolution3d_param();
			convolution_param->set_kernel_size(4);
			convolution_param->set_kernel_l(1);
			convolution_param->set_stride(2);
			convolution_param->set_num_output(2);
			convolution_param->mutable_weight_filler()->set_type("gaussian");
			convolution_param->mutable_weight_filler()->set_std(0.01);
			convolution_param->mutable_bias_filler()->set_type("constant");
			convolution_param->mutable_bias_filler()->set_value(1.);

			shared_ptr<Layer<Dtype>> layer(new Deconvolution3DLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 10);
			EXPECT_EQ(this->blob_top_->shape(4), 10);
		}
	};

	template <typename Dtype>
	class Pooling3DLayerTest {
	public:
		Pooling3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~Pooling3DLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(2, 2, 3, 10, 10));
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
		void StartTest(){
			LayerParameter layer_param;
			Pooling3DParameter* pooling_param = layer_param.mutable_pooling3d_param();
			pooling_param->set_kernel_size(4);
			pooling_param->set_kernel_l(1);
			pooling_param->set_stride(2);
			pooling_param->set_pool(Pooling3DParameter_PoolMethod_AVE);

			shared_ptr<Layer<Dtype>> layer(new Pooling3DLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 4);
			EXPECT_EQ(this->blob_top_->shape(4), 4);
		}
	};

	template <typename Dtype>
	class CropLayerTest {
	public:
		CropLayerTest() :blob_bottom1_(new Blob<Dtype>()),blob_bottom2_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~CropLayerTest(){ delete blob_bottom1_; delete blob_bottom2_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom1_->Reshape(video_shape(2, 2, 3, 10, 10));
			blob_bottom2_->Reshape(video_shape(1, 1, 2, 6, 7));
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom1_);
			filler.Fill(this->blob_bottom2_);
			blob_bottom_vec_.push_back(blob_bottom1_);
			blob_bottom_vec_.push_back(blob_bottom2_);
			blob_top_vec_.push_back(blob_top_);
		}
		Blob<Dtype>* const blob_bottom1_;
		Blob<Dtype>* const blob_bottom2_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;

	public:
		void StartTest(){
			LayerParameter layer_param;

			shared_ptr<Layer<Dtype>> layer(new CropLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 6);
			EXPECT_EQ(this->blob_top_->shape(4), 7);
		}
	};

	template <typename Dtype>
	class VideoSwitchLayerTest {
	public:
		VideoSwitchLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~VideoSwitchLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(6, 2, 10, 10);
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
		void StartTest(){
			LayerParameter layer_param;
			VideoSwitchParameter* video_switch_param = layer_param.mutable_video_switch_param();
			video_switch_param->set_switch_(VideoSwitchParameter_SwitchOp_VIDEO);
			video_switch_param->set_frame_num(3);

			shared_ptr<Layer<Dtype>> layer(new VideoSwitchLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 10);
			EXPECT_EQ(this->blob_top_->shape(4), 10);
		}
	};
}

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	caffe::Caffe::SetDevice(1);
	caffe::Convolution3DLayerTest<float> test1;
	test1.StartTest();
	LOG(INFO) << "End test Convolution3DLayer";
	caffe::Deconvolution3DLayerTest<float> test2;
	test2.StartTest();
	LOG(INFO) << "End test Deconvolution3DLayer";
	caffe::Pooling3DLayerTest<float> test3;
	test3.StartTest();
	LOG(INFO) << "End test Pooling3DLayer";
	caffe::CropLayerTest<float> test4;
	test4.StartTest();
	LOG(INFO) << "End test CropLayer";
	caffe::VideoSwitchLayerTest<float> test5;
	test5.StartTest();
	LOG(INFO) << "End test VideoSwitchLayer";
	return 0;
}
