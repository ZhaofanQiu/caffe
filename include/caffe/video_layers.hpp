
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

#include "leveldb/db.h"
#include "boost/scoped_ptr.hpp"

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
		virtual inline int MinNumTopBlobs() const { return 1; }
		virtual inline int MaxNumTopBlobs() const { return 2; }

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
		int filter_stride_, filter_stride_l_;
		int kernel_eff_, kernel_eff_l_;
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
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, filter_stride_, filter_stride_l_, col_buff);
		}
		inline void conv_col2vol_cpu(const Dtype* col_buff, Dtype* data) {
			col2vol_cpu(col_buff, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, filter_stride_, filter_stride_l_, data);
		}
#ifndef CPU_ONLY
		inline void conv_vol2col_gpu(const Dtype* data, Dtype* col_buff) {
			vol2col_gpu(data, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, filter_stride_, filter_stride_l_, col_buff);
		}
		inline void conv_col2vol_gpu(const Dtype* col_buff, Dtype* data) {
			col2vol_gpu(col_buff, conv_in_channels_, conv_in_length_, conv_in_height_, conv_in_width_,
				kernel_size_, kernel_l_, pad_, pad_l_, stride_, stride_l_, filter_stride_, filter_stride_l_, data);
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

		static Blob<Dtype> col_buffer_;
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

	template <typename Dtype>
	class VolumeDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit VolumeDataLayer(const LayerParameter& param);
		virtual ~VolumeDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// DataLayer uses DataReader instead for sharing for parallelism
		virtual inline bool ShareInParallel() const { return false; }
		virtual inline const char* type() const { return "VolumeData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		shared_ptr<leveldb::DB> db_;
		shared_ptr<leveldb::Iterator> iter_;
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual unsigned int PrefetchRand();
		virtual void load_batch(Batch<Dtype>* batch);
		vector<int> top_shape_;
		Blob<Dtype> data_mean_;
		int origin_width_;
		int origin_height_;
	};
	
	template <typename Dtype>
	class CropLayer : public Layer<Dtype> {
	public:
		explicit CropLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Crop"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int crop_h_, crop_w_;
	};

	template <typename Dtype>
	class VideoSwitchLayer : public Layer<Dtype> {
	public:
		explicit VideoSwitchLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoSwitch"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		bool to_video_;
		int frame_num_;
	};

	/**
	* @brief A helper for LSTMLayer: computes a single timestep of the
	*        non-linearity of the LSTM, producing the updated cell and hidden
	*        states.
	*/
	template <typename Dtype>
	class LSTMUnitLayer : public Layer<Dtype> {
	public:
		explicit LSTMUnitLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "LSTMUnit"; }

		virtual inline int MinNumBottomBlobs() const { return 2; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		/**
		* @param bottom input Blob vector (length, 2 * input_num)
		*   -# @f$ (1 \times N \times D) @f$
		*      the previous timestep cell state @f$ c_t-1 @f$
		*   -# @f$ (1 \times N \times 4D) @f$
		*      the "gate inputs" @f$ [i_t', f_t', o_t', g_t'] @f$
		* @param top output Blob vector (length, input_num * 2)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated cell state @f$ c_t @f$, computed as:
		*          i_t := \sigmoid[i_t']
		*          f_t := \sigmoid[f_t']
		*          o_t := \sigmoid[o_t']
		*          g_t := \tanh[g_t']
		*          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated hidden state @f$ h_t @f$, computed as:
		*          h_t := o_t .* \tanh[c_t]
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief The hidden and output dimension.
		int input_num_;
		int hidden_dim_;
		vector<shared_ptr<Blob<Dtype> > > X_acts_;
	};

	/**
	* @brief Implementation of Grid LSTM
	*/
	template <typename Dtype>
	class GridLSTMLayer : public Layer<Dtype> {
	public:
		explicit GridLSTMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GridLSTM"; }

		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		void CalculateOrder(const vector<Blob<Dtype>*>& bottom);
		/// @brief The hidden and output dimension.
		vector<bool> reverse_;
		int output_dim_;
		int num_seq_;
		int hidden_dim_;
		int grid_dim_;
		bool bias_term_;
		vector<vector<int> > link_idx_;
		vector<int> order_;
		//Data blobs
		shared_ptr<Blob<Dtype> > zero_memory_;

		//Layers
		// split_x_ layer
		shared_ptr<SplitLayer<Dtype> > split_x_;
		vector<shared_ptr<Blob<Dtype> > > X_;
		vector<shared_ptr<Blob<Dtype> > > X_1_;
		vector<shared_ptr<Blob<Dtype> > > X_2_;

		// split_h_ layer
		shared_ptr<SplitLayer<Dtype> > split_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_1_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_2_;
		
		// split_xh_h_ layer
		shared_ptr<SplitLayer<Dtype> > split_xh_h_;
		vector<shared_ptr<Blob<Dtype> > > XH_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > XH_h_k_;

		// concat_h_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_h_;

		// concat_x_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_x_;
		vector<shared_ptr<Blob<Dtype> > > XH_x_;

		// ip_xh_h_ layer
		vector<shared_ptr<InnerProductLayer<Dtype> > > ip_xh_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > G_h_;

		// lstm_unit_h_ layer
		shared_ptr<LSTMUnitLayer<Dtype> > lstm_unit_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > C_i_;
	};

	/**
	* @brief Normalizes input.
	* https://github.com/kuprel/caffe
	*/
	template <typename Dtype>
	class NormalizeLayer : public Layer<Dtype> {
	public:
		explicit NormalizeLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "Normalize"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> norm_;
		Blob<Dtype> sum_channel_multiplier_, sum_spatial_multiplier_;
		Blob<Dtype> buffer_, buffer_channel_, buffer_spatial_;
		bool across_spatial_;
		bool channel_shared_;
		Dtype eps_;
	};

	/**
	* @brief UnPools the input image by assigning fixed, bilinear interpolation,
	* etc. within regions.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class UnPoolingLayer : public Layer<Dtype> {
	public:
		explicit UnPoolingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "UnPooling"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		// fill mask for different unpool type
		void FillMask();

		int out_kernel_h_, out_kernel_w_;
		int out_stride_h_, out_stride_w_;
		int out_pad_h_, out_pad_w_;
		int num_, channels_;
		int height_, width_;
		int unpooled_height_, unpooled_width_;
		Blob<int> mask_;
	};

	/**
	* @brief Count the prediction and ground truth statistics for each datum.
	*
	* NOTE: This does not implement Backwards operation.
	*/
	template <typename Dtype>
	class ParseEvaluateLayer : public Layer<Dtype> {
	public:
		/**
		* @param param provides ParseEvaluateParameter parse_evaluate_param,
		*     with ParseEvaluateLayer options:
		*   - num_labels (\b optional int32.).
		*     number of labels. must provide!!
		*   - ignore_label (\b repeated int32).
		*     If any, ignore evaluating the corresponding label for each
		*     image.
		*/
		explicit ParseEvaluateLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ParseEvaluate"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		/**
		* @param bottom input Blob vector (length 2)
		*   -# @f$ (N \times 1 \times H \times W) @f$
		*      the prediction label @f$ x @f$
		*   -# @f$ (N \times 1 \times H \times W) @f$
		*      the ground truth label @f$ x @f$
		* @param top output Blob vector (length 1)
		*   -# @f$ (N \times C \times 1 \times 3) @f$
		*      the counts for different class @f$
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/// @brief Not implemented (non-differentiable function)
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
		}

		// number of total labels
		int num_labels_;
		// store ignored labels
		std::set<Dtype> ignore_labels_;
	};

	/**
	* @brief Compute the segmentation label of the @f$ H \times W @f$ for each datum across
	*        all channels @f$ C @f$.
	*
	* Intended for use after a classification layer to produce a prediction of
	* segmentation label.
	* If parameter out_max_val is set to true, also output the predicted value for
	* the corresponding label for each image.
	*
	* NOTE: does not implement Backwards operation.
	*/
	template <typename Dtype>
	class ParseOutputLayer : public Layer<Dtype> {
	public:
		/**
		* @param param provides ParseOutputParameter parse_output_param,
		*     with ParseOutputLayer options:
		*   - out_max_val (\b optional bool, default false).
		*     if set, output the predicted value for the corresponding label for each
		*     image.
		*/
		explicit ParseOutputLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ParseOutput"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		/**
		* @param bottom input Blob vector (length 1)
		*   -# @f$ (N \times C \times H \times W) @f$
		*      the inputs @f$ x @f$
		* @param top output Blob vector (length 1)
		*   -# @f$ (N \times 1 \times H \times W) @f$ or, if out_max_val
		*      @f$ (N \times 2 \times H \times W) @f$
		*      the computed outputs @f$
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		/// @brief Not implemented (non-differentiable function)
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
			NOT_IMPLEMENTED;
		}

		bool out_max_val_;

		// max_prob_ is used to store the maximum probability value
		Blob<Dtype> max_prob_;
	};

	template <typename Dtype>
	class ExtractFrameLayer : public Layer<Dtype> {
	public:
		explicit ExtractFrameLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ExtractFrame"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int frame_id_;
	};	

	template <typename Dtype>
	class RandomFusionLayer : public Layer<Dtype> {
	public:
		explicit RandomFusionLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "RandomFusion"; }
		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		vector<Dtype> random_vec_;
		vector<unsigned int> random_idx_;
		float std_;
		float mean_;
		float prob_;
	};

	template <typename Dtype>
	class RandomLossLayer : public LossLayer<Dtype> {
	public:
		explicit RandomLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "RandomLoss"; }
		virtual inline int ExactNumTopBlobs() const { return -1; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


		/// The internal SoftmaxLayer used to map predictions to a distribution.
		shared_ptr<Layer<Dtype> > softmax_layer_;
		/// prob stores the output probability predictions from the SoftmaxLayer.
		Blob<Dtype> prob_;
		/// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
		vector<Blob<Dtype>*> softmax_bottom_vec_;
		/// top vector holder used in call to the underlying SoftmaxLayer::Forward
		vector<Blob<Dtype>*> softmax_top_vec_;
		/// Whether to ignore instances with a certain label.
		bool has_ignore_label_;
		/// The label indicating that an instance should be ignored.
		int ignore_label_;
		/// Whether to normalize the loss by the total number of values present
		/// (otherwise just by the batch size).
		bool normalize_;

		int softmax_axis_, outer_num_, inner_num_;

		Blob<unsigned int> random_idx_;
		int num_label_;
		float dropout_ratio_;
		int uint_thres_;
	};
}  // namespace caffe

#endif  // CAFFE_VIDEO_LAYERS_HPP_
