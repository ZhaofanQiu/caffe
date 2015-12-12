
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/2
** desc： EncodeMachineLayer layer
*********************************************************************************/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

template <typename Dtype>
void EncodeMachineLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	EncodeMachineParameter param = this->layer_param_.encode_machine_param();
	CHECK(param.has_net() &&
		param.has_encode_begin() &&
		param.has_encode_end() &&
		param.has_decode_begin() &&
		param.has_decode_end()) << "Miss some encode machine parameter";

	// encode net parameter
	net_file_ = param.net();
	encode_begin_ = param.encode_begin();
	encode_end_ = param.encode_end();
	decode_begin_ = param.decode_begin();
	decode_end_ = param.decode_end();

	// sample parameter
	cd_k_ = param.cd_k();
	s_k_ = param.sample_k();

	// update parameter
	loss_weight_ = param.loss_weight();

	// init encode net
	net_.reset(new Net<Dtype>(net_file_, this->phase_));
	
	// link pv_, ph_
	vis_blob_ = net_->blob_by_name("vis_blob");
	hid_blob_ = net_->blob_by_name("hid_blob");
	re_vis_blob_ = net_->blob_by_name("re_vis_blob");
	count_v_ = vis_blob_->count();
	count_h_ = hid_blob_->count();
	CHECK_EQ(vis_blob_->count(), re_vis_blob_->count());

	// share parameters
	vector<shared_ptr<Blob<Dtype> > > net_params = net_->params();
	this->blobs_.clear();
	this->blobs_.insert(blobs_.end(), net_params.begin(), net_params.end());

	// init temp blobs
	v0_.reset(new Blob<Dtype>());
	h0_.reset(new Blob<Dtype>());
	mean_h0_.reset(new Blob<Dtype>());
	vk_.reset(new Blob<Dtype>());
	hk_.reset(new Blob<Dtype>());
	sample_v_.reset(new Blob<Dtype>());
	sample_h_.reset(new Blob<Dtype>());
	diff_v_.reset(new Blob<Dtype>());
}

template <typename Dtype>
void EncodeMachineLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top)
{
	CHECK(bottom[0]->count(), vis_blob_->count());
	top[0]->Reshape(hid_blob_->shape());
	top[1]->Reshape(vector<int>(0));

	this->v0_->Reshape(vis_blob_->shape());
	this->vk_->Reshape(vis_blob_->shape());
	this->sample_v_->Reshape(vis_blob_->shape());
	this->h0_->Reshape(hid_blob_->shape());
	this->mean_h0_->Reshape(hid_blob_->shape());
	this->hk_->Reshape(hid_blob_->shape());
	this->sample_h_->Reshape(hid_blob_->shape());
	this->diff_v_->Reshape(vis_blob_->shape());
}


template <typename Dtype>
void SampleAct(int count, const Dtype* input, const Dtype* noise, Dtype* output)
{
	for (int i = 0; i < count; ++i)
	{
		output[i] = max<Dtype>(input[i] + noise[i] * sqrt(1. / (1 + exp(-input[i]))), 0);
	}
}

template <typename Dtype>
void EncodeMachineLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// share parameters
	vector<shared_ptr<Blob<Dtype> > > net_params = net_->params();
	for (int i = 0; i < net_params.size(); ++i)
	{
		net_params[i]->ShareData(*blobs_[i]);
		net_params[i]->ShareDiff(*blobs_[i]);
	}

	// copy bottom[0] -> v0_, bottom[0] -> vis_blob_
	caffe_copy(count_v_, bottom[0]->cpu_data(), v0_->mutable_cpu_data());
	caffe_copy(count_v_, bottom[0]->cpu_data(), vis_blob_->mutable_cpu_data());

	// init encode, h0_ -> top[0]
	net_->ForwardFromTo(encode_begin_, encode_end_);
	caffe_copy(count_h_, hid_blob_->cpu_data(), top[0]->mutable_cpu_data());
	caffe_copy(count_h_, hid_blob_->cpu_data(), mean_h0_->mutable_cpu_data());

	if (this->phase_ == TRAIN)
	{
		// sample hid_blob_
		caffe_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_cpu_data());
		SampleAct(count_h_, hid_blob_->cpu_data(), sample_h_->cpu_data(), hid_blob_->mutable_cpu_data());

		// 1~cd_k decode/encode
		for (int k = 0; k < cd_k_; k++)
		{
			// decode hk-1 -> vk
			net_->ForwardFromTo(decode_begin_, decode_end_);
			// sample vk -> vk
			caffe_rng_gaussian<Dtype>(count_v_, 0, 1, sample_v_->mutable_cpu_data());
			SampleAct(count_v_, re_vis_blob_->cpu_data(), sample_v_->cpu_data(), re_vis_blob_->mutable_cpu_data());

			if (k == 0)
			{
				// calculate loss
				caffe_sub(count_v_, v0_->cpu_data(), vis_blob_->cpu_data(), diff_v_->mutable_cpu_data());
				Dtype dot = caffe_cpu_dot(count_v_, diff_v_->cpu_data(), diff_v_->cpu_data());
				Dtype loss = loss_weight_ * dot / bottom[0]->shape(0) / Dtype(2);
				top[1]->mutable_cpu_data()[0] = loss;
			}

			// encode vk -> hk
			caffe_copy(count_v_, re_vis_blob_->cpu_data(), vis_blob_->mutable_cpu_data());
			net_->ForwardFromTo(encode_begin_, encode_end_);
			// sample hk -> hk
			caffe_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_cpu_data());
			SampleAct(count_h_, hid_blob_->cpu_data(), sample_h_->cpu_data(), hid_blob_->mutable_cpu_data());

		}
		caffe_copy(count_h_, hid_blob_->cpu_data(), hk_->mutable_cpu_data());
		caffe_copy(count_v_, vis_blob_->cpu_data(), vk_->mutable_cpu_data());
	}
}

template <typename Dtype>
void EncodeMachineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// share parameters
	vector<shared_ptr<Blob<Dtype> > > net_params = net_->params();
	for (int i = 0; i < net_params.size(); ++i)
	{
		net_params[i]->ShareData(*blobs_[i]);
		net_params[i]->ShareDiff(*blobs_[i]);
	}

	if (this->phase_ == TRAIN)
	{
		// Grad from top
		caffe_copy(count_v_, bottom[0]->cpu_data(), vis_blob_->mutable_cpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_copy(count_h_, top[0]->cpu_diff(), hid_blob_->mutable_cpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);

		// E<p(h, v)> 1
		caffe_copy(count_h_, hk_->cpu_data(), hid_blob_->mutable_cpu_data());
		net_->ForwardFromTo(decode_begin_, decode_end_);

		caffe_set(count_v_, Dtype(0.), re_vis_blob_->mutable_cpu_diff());
		caffe_axpy(count_v_, loss_weight_, vk_->cpu_data(), re_vis_blob_->mutable_cpu_diff());
		net_->BackwardFromTo(decode_end_, decode_begin_);
		// E<p(h, v)> 2
		caffe_copy(count_v_, vk_->cpu_data(), vis_blob_->mutable_cpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_set(count_h_, Dtype(0.), hid_blob_->mutable_cpu_diff());
		caffe_axpy(count_h_, loss_weight_, hk_->cpu_data(), hid_blob_->mutable_cpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);
	}

	if (s_k_ == 0)
	{
		// no sample h0
		caffe_set(count_h_, Dtype(0.), sample_h_->mutable_cpu_data());
		SampleAct(count_h_, mean_h0_->cpu_data(), sample_h_->cpu_data(), h0_->mutable_cpu_data());

		// E<p(h|v)> 1
		caffe_copy(count_h_, h0_->cpu_data(), hid_blob_->mutable_cpu_data());
		net_->ForwardFromTo(decode_begin_, decode_end_);

		caffe_set(count_v_, Dtype(0.), re_vis_blob_->mutable_cpu_diff());
		caffe_axpy(count_v_, -loss_weight_, v0_->cpu_data(), re_vis_blob_->mutable_cpu_diff());
		net_->BackwardFromTo(decode_end_, decode_begin_);
		// E<p(h|v)> 2
		caffe_copy(count_v_, v0_->cpu_data(), vis_blob_->mutable_cpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_set(count_h_, Dtype(0.), hid_blob_->mutable_cpu_diff());
		caffe_axpy(count_h_, -loss_weight_, h0_->cpu_data(), hid_blob_->mutable_cpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);
	}
	else
	{
		for (int k = 0; k < s_k_; ++k)
		{
			// sample h0
			caffe_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_cpu_data());
			SampleAct(count_h_, mean_h0_->cpu_data(), sample_h_->cpu_data(), h0_->mutable_cpu_data());

			// E<p(h|v)> 1
			caffe_copy(count_h_, h0_->cpu_data(), hid_blob_->mutable_cpu_data());
			net_->ForwardFromTo(decode_begin_, decode_end_);

			caffe_set(count_v_, Dtype(0.), re_vis_blob_->mutable_cpu_diff());
			caffe_axpy(count_v_, -loss_weight_ / s_k_, v0_->cpu_data(), re_vis_blob_->mutable_cpu_diff());
			net_->BackwardFromTo(decode_end_, decode_begin_);
			// E<p(h|v)> 2
			caffe_copy(count_v_, v0_->cpu_data(), vis_blob_->mutable_cpu_data());
			net_->ForwardFromTo(encode_begin_, encode_end_);

			caffe_set(count_h_, Dtype(0.), hid_blob_->mutable_cpu_diff());
			caffe_axpy(count_h_, -loss_weight_ / s_k_, h0_->cpu_data(), hid_blob_->mutable_cpu_diff());
			net_->BackwardFromTo(encode_end_, encode_begin_);
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(EncodeMachineLayer);
#endif

INSTANTIATE_CLASS(EncodeMachineLayer);
REGISTER_LAYER_CLASS(EncodeMachine);

}  // namespace caffe
