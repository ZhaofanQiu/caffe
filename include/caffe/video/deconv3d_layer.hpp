
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： Deconv3D layer
*********************************************************************************/

#ifndef CAFFE_DECONV3D_LAYER_HPP_
#define CAFFE_DECONV3D_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/base_conv3d_layer.hpp"

namespace caffe {
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
}  // namespace caffe

#endif  // CAFFE_DECONV3D_LAYER_HPP_
