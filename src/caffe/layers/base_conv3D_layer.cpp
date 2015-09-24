
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/11
** desc： BaseConvolution3D layer
*********************************************************************************/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	Blob<Dtype> BaseConvolution3DLayer<Dtype>::col_buffer_;

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
      << "corresponding to (num, channels, length, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  Convolution3DParameter conv_param = this->layer_param_.convolution3d_param();
  CHECK(conv_param.has_kernel_l())
	  << "Filter length kernel_l is required.";
  CHECK(conv_param.has_kernel_size())
      << "Filter kernel_size is required.";
  kernel_l_ = conv_param.kernel_l();
  kernel_size_ = conv_param.kernel_size();
  CHECK_GT(kernel_l_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_size_, 0) << "Filter dimensions cannot be zero.";
  pad_l_ = conv_param.pad_l();
  pad_ = conv_param.pad();
  stride_l_ = conv_param.stride_l();
  stride_ = conv_param.stride();
  filter_stride_ = conv_param.filter_stride();
  filter_stride_l_ = conv_param.filter_stride_l();
  kernel_eff_ = kernel_size_ + (kernel_size_ - 1) * (filter_stride_ - 1);
  kernel_eff_l_ = kernel_l_ + (kernel_l_ - 1) * (filter_stride_l_ - 1);

  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_l_ == 1 && kernel_size_ == 1
	  && filter_stride_ == 1 && filter_stride_l_ == 1
      && stride_l_ == 1 && stride_ && pad_l_ == 0 && pad_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(1);
  num_output_ = this->layer_param_.convolution3d_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution3d_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution3d_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
	this->blobs_[0].reset(new Blob<Dtype>(video_shape(
        conv_out_channels_, conv_in_channels_ / group_, kernel_l_, kernel_size_, kernel_size_)));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution3d_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution3d_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 5 axes, "
      << "corresponding to (num, channels, length, height, width)";
  num_ = bottom[0]->shape(0);
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  CHECK_EQ(bottom[0]->shape(1), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->shape(0)) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->shape(1))
        << "Inputs must have same channels.";
    CHECK_EQ(length_, bottom[bottom_id]->shape(2))
        << "Inputs must have same length.";
    CHECK_EQ(height_, bottom[bottom_id]->shape(3))
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->shape(4))
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  vector<int> top_shape = video_shape(num_, num_output_, length_out_, height_out_, width_out_);
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
	  conv_in_length_ = length_out_;
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = length_ * height_ * width_;
  } else {
	  conv_in_length_ = length_;
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = length_out_ * height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_l_ * kernel_size_ * kernel_size_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  vector<int> col_shape = vector<int>(5, 0);
  col_shape[0] = 1;
  col_shape[1] = kernel_dim_;
  if (reverse_dimensions()) {
	  col_shape[2] = length_;
	  col_shape[3] = height_;
	  col_shape[4] = width_;
  } else {
	  col_shape[2] = length_out_;
	  col_shape[3] = height_out_;
	  col_shape[4] = width_out_;
  }
  col_buffer_.Reshape(col_shape);
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, length_out_ * height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_vol2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_vol2col) {
      conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      length_out_ * height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2vol_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, length_out_ * height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_vol2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_vol2col) {
      conv_vol2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      length_out_ * height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2vol_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_vol2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolution3DLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, length_out_ * height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY
INSTANTIATE_CLASS(BaseConvolution3DLayer);
}  // namespace caffe
