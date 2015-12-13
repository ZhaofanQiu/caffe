
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： Conv3D layer
*********************************************************************************/

#ifndef CAFFE_CONV3D_LAYER_HPP_
#define CAFFE_CONV3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/base_conv3d_layer.hpp"

namespace caffe {
	/**
	* @brief Convolves the input video with a bank of learned filters,
	*        and (optionally) adds biases.
	*
	*   Caffe convolves by reduction to matrix multiplication. This achieves
	*   high-throughput and generality of input and filter dimensions but comes at
	*   the cost of memory for matrices. This makes use of efficiency in BLAS.
	*
	*   The input is "vol2col" transformed to a channel K' x L x H x W data matrix
	*   for multiplication with the N x K' x L x H x W filter matrix to yield a
	*   N' x L x H x W output matrix that is then "col2vol" restored. K' is the
	*   input channel * kernel length * height * kernel width dimension of the unrolled
	*   inputs so that the vol2col matrix has a column for each input region to
	*   be filtered. col2vol restores the output spatial structure by rolling up
	*   the output channel N' columns of the output matrix.
	*/
	template <typename Dtype>
	class Convolution3DLayer : public BaseConvolution3DLayer<Dtype> {
	public:
		/**
		* @param param provides ConvolutionParameter convolution_param,
		*    with ConvolutionLayer options:
		*  - num_output. The number of filters.
		*  - kernel_size. The filter dimensions, given by
		*  kernel_size for square filters.
		*  - stride (\b optional, default 1). The filter
		*  stride, given by stride_size for equal dimensions.
		*  By default the convolution is dense with stride 1.
		*  - pad (\b optional, default 0). The zero-padding for
		*  convolution, given by pad for equal dimensions or pad_h and pad_w for
		*  different padding. Input padding is computed implicitly instead of
		*  actually padding.
		*  - group (\b optional, default 1). The number of filter groups. Group
		*  convolution is a method for reducing parameterization by selectively
		*  connecting input and output channels. The input and output channel dimensions must be divisible
		*  by the number of groups. For group @f$ \geq 1 @f$, the
		*  convolutional filters' input and output channels are separated s.t. each
		*  group takes 1 / group of the input channels and makes 1 / group of the
		*  output channels. Concretely 4 input channels, 8 output channels, and
		*  2 groups separate input channels 1-2 and output channels 1-4 into the
		*  first group and input channels 3-4 and output channels 5-8 into the second
		*  group.
		*  - bias_term (\b optional, default true). Whether to have a bias.
		*  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
		*    kernels + stream parallelism) engines.
		*/
		explicit Convolution3DLayer(const LayerParameter& param)
			: BaseConvolution3DLayer<Dtype>(param) {}

		virtual inline const char* type() const { return "Convolution3D"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual inline bool reverse_dimensions() { return false; }
		virtual void compute_output_shape();
	};
}  // namespace caffe

#endif  // CAFFE_CONV3D_LAYER_HPP_
