
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
			convolution_param->set_kernel_size(3);
			convolution_param->set_kernel_l(2);
			convolution_param->set_stride(2);
			convolution_param->set_filter_stride(2);
			convolution_param->set_filter_stride_l(2);
			convolution_param->set_pad(2);
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
			EXPECT_EQ(this->blob_top_->shape(2), 1);
			EXPECT_EQ(this->blob_top_->shape(3), 5);
			EXPECT_EQ(this->blob_top_->shape(4), 5);
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
			pooling_param->set_pool(Pooling3DParameter_PoolMethod_MAX);

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
			VideoSwitchParameter* video_switch_param = layer_param.mutable_video_switch_param();
			video_switch_param->set_switch_(VideoSwitchParameter_SwitchOp_IMAGE);

			shared_ptr<Layer<Dtype>> layer(new VideoSwitchLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 6);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 10);
			EXPECT_EQ(this->blob_top_->shape(3), 10);
		}
	};

	template <typename Dtype>
	class LSTMUnitLayerTest {
	public:
		LSTMUnitLayerTest() :blob_bottom1_(new Blob<Dtype>()), blob_bottom2_(new Blob<Dtype>()),
			blob_top1_(new Blob<Dtype>()), blob_top2_(new Blob<Dtype>()){
			this->SetUp();
		}
		~LSTMUnitLayerTest(){ delete blob_bottom1_; delete blob_bottom2_; delete blob_top1_; delete blob_top2_; }

	protected:
		void SetUp(){
			vector<int> shape1 = vector<int>(2, 0);
			shape1[0] = 3; shape1[1] = 4;
			blob_bottom1_->Reshape(shape1);

			vector<int> shape2 = vector<int>(2, 0);
			shape2[0] = 3; shape2[1] = 16;
			blob_bottom2_->Reshape(shape2);

			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom1_);
			filler.Fill(this->blob_bottom2_);
			blob_bottom_vec_.push_back(blob_bottom1_);
			blob_bottom_vec_.push_back(blob_bottom2_);
			blob_top_vec_.push_back(blob_top1_);
			blob_top_vec_.push_back(blob_top2_);
		}
		Blob<Dtype>* const blob_bottom1_;
		Blob<Dtype>* const blob_bottom2_;
		Blob<Dtype>* const blob_top1_;
		Blob<Dtype>* const blob_top2_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;

	public:
		void StartTest(){
			LayerParameter layer_param;

			shared_ptr<Layer<Dtype>> layer(new LSTMUnitLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top1_->shape(0), 3);
			EXPECT_EQ(this->blob_top1_->shape(1), 4);
			EXPECT_EQ(this->blob_top2_->shape(0), 3);
			EXPECT_EQ(this->blob_top2_->shape(1), 4);
		}
	};

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
			blob_bottom_->Reshape(video_shape(2, 3, 3, 2, 3));
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
			InnerProductParameter* ip_param = layer_param.mutable_inner_product_param();
			GridLSTMParameter* grid_param = layer_param.mutable_grid_lstm_param();
			ip_param->set_num_output(2);
			FillerParameter* fp1 = ip_param->mutable_weight_filler();
			fp1->set_type("gaussian");
			fp1->set_std(0.01);
			FillerParameter* fp2 = ip_param->mutable_bias_filler();
			fp2->set_type("constant");
			fp2->set_value(0);
			grid_param->add_reverse(true);
			grid_param->add_reverse(true);
			grid_param->add_reverse(true);

			shared_ptr<Layer<Dtype>> layer(new GridLSTMLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 9);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 2);
			EXPECT_EQ(this->blob_top_->shape(4), 3);
		}
	};

	template <typename Dtype>
	class ExtractFrameLayerTest {
	public:
		ExtractFrameLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~ExtractFrameLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(3, 2, 4, 10, 10));
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
			ExtractFrameParameter* extract_frame_param = layer_param.mutable_extract_frame_param();
			extract_frame_param->set_frame(2);

			shared_ptr<Layer<Dtype>> layer(new ExtractFrameLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-3, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 3);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 10);
			EXPECT_EQ(this->blob_top_->shape(3), 10);
		}
	};

	template <typename Dtype>
	class RandomFusionLayerTest {
	public:
		RandomFusionLayerTest() :blob_bottom1_(new Blob<Dtype>()), blob_bottom2_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~RandomFusionLayerTest(){ delete blob_bottom1_; delete blob_bottom2_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom1_->Reshape(video_shape(2, 2, 3, 3, 10));
			blob_bottom2_->Reshape(video_shape(2, 2, 3, 3, 10));
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
	class MapLSTMUnitLayerTest {
	public:
		MapLSTMUnitLayerTest() :blob_bottom1_(new Blob<Dtype>()), blob_bottom2_(new Blob<Dtype>()),
			blob_top1_(new Blob<Dtype>()), blob_top2_(new Blob<Dtype>()){
			this->SetUp();
		}
		~MapLSTMUnitLayerTest(){ delete blob_bottom1_; delete blob_bottom2_; delete blob_top1_; delete blob_top2_; }

	protected:
		void SetUp(){
			blob_bottom1_->Reshape(2, 2, 4, 5);
			blob_bottom2_->Reshape(2, 8, 4, 5);

			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom1_);
			filler.Fill(this->blob_bottom2_);
			blob_bottom_vec_.push_back(blob_bottom1_);
			blob_bottom_vec_.push_back(blob_bottom2_);
			blob_top_vec_.push_back(blob_top1_);
			blob_top_vec_.push_back(blob_top2_);
		}
		Blob<Dtype>* const blob_bottom1_;
		Blob<Dtype>* const blob_bottom2_;
		Blob<Dtype>* const blob_top1_;
		Blob<Dtype>* const blob_top2_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;

	public:
		void StartTest(){
			LayerParameter layer_param;

			shared_ptr<Layer<Dtype>> layer(new MapLSTMUnitLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top1_->shape(0), 2);
			EXPECT_EQ(this->blob_top1_->shape(1), 2);
			EXPECT_EQ(this->blob_top1_->shape(2), 4);
			EXPECT_EQ(this->blob_top1_->shape(3), 5);
			EXPECT_EQ(this->blob_top2_->shape(0), 2);
			EXPECT_EQ(this->blob_top2_->shape(1), 2);
			EXPECT_EQ(this->blob_top2_->shape(2), 4);
			EXPECT_EQ(this->blob_top2_->shape(3), 5);
		}
	};

	template <typename Dtype>
	class MapLSTMLayerTest {
	public:
		MapLSTMLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~MapLSTMLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(video_shape(2, 3, 3, 2, 3));
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
			ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
			conv_param->set_num_output(2);
			conv_param->set_kernel_size(3);
			conv_param->set_pad(1);
			FillerParameter* fp1 = conv_param->mutable_weight_filler();
			fp1->set_type("gaussian");
			fp1->set_std(0.01);
			FillerParameter* fp2 = conv_param->mutable_bias_filler();
			fp2->set_type("constant");
			fp2->set_value(0);

			shared_ptr<Layer<Dtype>> layer(new MapLSTMLayer<Dtype>(layer_param));
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->shape(0), 2);
			EXPECT_EQ(this->blob_top_->shape(1), 2);
			EXPECT_EQ(this->blob_top_->shape(2), 3);
			EXPECT_EQ(this->blob_top_->shape(3), 2);
			EXPECT_EQ(this->blob_top_->shape(4), 3);
		}
	};
}

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	//caffe::Caffe::set_mode(caffe::Caffe::CPU);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
	caffe::Caffe::SetDevice(10);
	/*
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
	caffe::LSTMUnitLayerTest<float> test6;
	test6.StartTest();
	LOG(INFO) << "End test LSTMUnitLayer";
	caffe::GridLSTMLayerTest<float> test7;
	test7.StartTest();
	LOG(INFO) << "End test GridLSTMLayer";
	caffe::ExtractFrameLayerTest<float> test8;
	test8.StartTest();
	LOG(INFO) << "End test ExtractFrameLayer";
	*/
	caffe::MapLSTMUnitLayerTest<float> test9;
	test9.StartTest();
	LOG(INFO) << "End test MapLSTMUnitLayer";
	//caffe::MapLSTMLayerTest<float> test10;
	//test10.StartTest();
	//LOG(INFO) << "End test MapLSTMLayer";
	return 0;
}
