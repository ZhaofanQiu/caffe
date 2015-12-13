
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： EncodeMachine layer
*********************************************************************************/

#ifndef CAFFE_ENCODE_MACHINE_LAYER_HPP_
#define CAFFE_ENCODE_MACHINE_LAYER_HPP_

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

namespace caffe {
	template <typename Dtype>
	class EncodeMachineLayer : public Layer<Dtype> {
	public:
		explicit EncodeMachineLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {
			}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "EncodeMachine"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		string net_file_;
		shared_ptr<Net<Dtype> > net_;
		int encode_begin_, encode_end_, decode_begin_, decode_end_;
		Dtype loss_weight_;
		int cd_k_, s_k_;

		int count_v_, count_h_;
		shared_ptr<Blob<Dtype> > vis_blob_, hid_blob_, re_vis_blob_;
		shared_ptr<Blob<Dtype> > v0_, mean_h0_, h0_, vk_, hk_;
		shared_ptr<Blob<Dtype> > sample_v_, sample_h_;
		shared_ptr<Blob<Dtype> > diff_v_;
	};
}  // namespace caffe

#endif  // CAFFE_ENCODE_MACHINE_LAYER_HPP_
