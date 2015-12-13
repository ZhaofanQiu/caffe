
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： BaseConv3D layer header
*********************************************************************************/

#ifndef CAFFE_BASE_CONV3D_LAYER_HPP_
#define CAFFE_BASE_CONV3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/video/video_common.hpp"
#include "caffe/util/vol2col.hpp"

namespace caffe {
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

		Blob<Dtype> bias_multiplier_;
		static Blob<Dtype> col_buffer_;
		//Blob<Dtype> col_buffer_;
		int conv_out_spatial_dim_;
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
		int conv_in_length_;
		int conv_in_height_;
		int conv_in_width_;
		int kernel_dim_;
		int weight_offset_; 
		int col_offset_;
		int output_offset_;
	};

}  // namespace caffe

#endif  // CAFFE_BASE_CONV3D_LAYER_HPP_
