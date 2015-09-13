
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/3
** desc： Videodata layer
*********************************************************************************/

#ifndef CAFFE_VIDEO_LAYERS_HPP_
#define CAFFE_VIDEO_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	vector<int> video_shape(int num, int channels = 0, int length = 0, int height = 0, int width = 0);

	/**
	* @brief Layer to read video data from list.
	*/
	template <typename Dtype>
	class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit VideoDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~VideoDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual void load_batch(Batch<Dtype>* batch);
		virtual void ShuffleVideo();
		virtual unsigned int PrefetchRand();

		vector<string> file_list_;
		vector<int> start_frm_list_;
		vector<int> label_list_;
		vector<int> shuffle_index_;
		int lines_id_;
		vector<int> top_shape_;

		Blob<Dtype> data_mean_;
		int origin_width_;
		int origin_height_;
	};

	/**
	* @brief Abstract base class that factors out the BLAS code common to
	*        Convolution3DLayer and Deconvolution3DLayer.
	*/
	template <typename Dtype>
	class BaseConvolution3DLayer : public Layer<Dtype> {
	public:
		explicit BaseConvolution3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		// Helper functions that abstract away the column buffer and gemm arguments.
		// The last argument in forward_cpu_gemm is so that we can skip the im2col if
		// we just called weight_cpu_gemm with the same input.
		void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* output, bool skip_im2col = false);
		void forward_cpu_bias(Dtype* output, const Dtype* bias);
		void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* output);
		void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
			weights);
		void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
		void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
			Dtype* output, bool skip_im2col = false);
		void forward_gpu_bias(Dtype* output, const Dtype* bias);
		void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
			Dtype* col_output);
		void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
			weights);
		void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

		// reverse_dimensions should return true iff we are implementing deconv, so
		// that conv helpers know which dimensions are which.
		virtual bool reverse_dimensions() = 0;
		// Compute height_out_ and width_out_ from other parameters.
		virtual void compute_output_shape() = 0;

		int kernel_l_, kernel_size_;
		int stride_l_, stride_;
		int num_;
		int channels_;
		int pad_l_, pad_;
		int length_, height_, width_;
		int group_;
		int num_output_;
		int length_out_, height_out_, width_out_;
		bool bias_term_;
		bool is_1x1_;

	private:
		// wrap im2col/col2im so we don't have to remember the (long) argument lists
		inline void conv_vol2col_cpu(const Dtype* data, Dtype* col_buff) {
			vol2col_cpu(data, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, col_buff);
		}
		inline void conv_col2vol_cpu(const Dtype* col_buff, Dtype* data) {
			col2vol_cpu(col_buff, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, data);
		}
#ifndef CPU_ONLY
		inline void conv_vol2col_gpu(const Dtype* data, Dtype* col_buff) {
			vol2col_gpu(data, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, col_buff);
		}
		inline void conv_col2vol_gpu(const Dtype* col_buff, Dtype* data) {
			col2vol_gpu(col_buff, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, data);
		}
#endif

		int conv_out_channels_;
		int conv_in_channels_;
		int conv_out_spatial_dim_;
		int conv_in_length_;
		int conv_in_height_;
		int conv_in_width_;
		int kernel_dim_;
		int weight_offset_;
		int col_offset_;
		int output_offset_;

		Blob<Dtype> col_buffer_;
		Blob<Dtype> bias_multiplier_;
	};

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

	template <typename Dtype>
	class Deconvolution3DLayer : public BaseConvolution3DLayer<Dtype> {
	public:
		explicit Deconvolution3DLayer(const LayerParameter& param)
			: BaseConvolution3DLayer<Dtype>(param) {}

		virtual inline const char* type() const { return "Deconvolution3D"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual inline bool reverse_dimensions() { return true; }
		virtual void compute_output_shape();
	};

	/**
	* @brief Pools the input video by taking the max, average, etc. within regions.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class Pooling3DLayer : public Layer<Dtype> {
	public:
		explicit Pooling3DLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Pooling3D"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MinTopBlobs() const { return 1; }
		// MAX POOL layers can output an extra top blob for the mask;
		// others can only output the pooled inputs.
		virtual inline int MaxTopBlobs() const {
			return (this->layer_param_.pooling_param().pool() ==
				PoolingParameter_PoolMethod_MAX) ? 2 : 1;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int kernel_l_, kernel_h_, kernel_w_;
		int stride_l_, stride_h_, stride_w_;
		int pad_l_, pad_h_, pad_w_;
		int channels_;
		int length_, height_, width_;
		int pooled_length_, pooled_height_, pooled_width_;
		bool global_pooling_;
		Blob<Dtype> rand_idx_;
		Blob<int> max_idx_;
	};
}  // namespace caffe

#endif  // CAFFE_VIDEO_LAYERS_HPP_
