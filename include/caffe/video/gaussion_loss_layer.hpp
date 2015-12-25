
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/20
** desc： GaussionLoss layer
*********************************************************************************/

#ifndef CAFFE_GAUSSION_LOSS_LAYER_HPP_
#define CAFFE_GAUSSION_LOSS_LAYER_HPP_

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

#include "caffe/layers/loss_layer.hpp"
#include "caffe/video/video_common.hpp"

namespace caffe {
	template <typename Dtype>
	class GaussionLossLayer : public LossLayer<Dtype> {
	public:
		explicit GaussionLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {
			}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

		virtual inline const char* type() const { return "GaussionLoss"; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int N_;
		Blob<Dtype> diff_;
		Dtype eps_;
	};
}  // namespace caffe

#endif  // CAFFE_GAUSSION_LOSS_LAYER_HPP_
