
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
	__global__ void SampleAct(const int count, const Dtype* input, const Dtype* noise,
		Dtype* output) {
		CUDA_KERNEL_LOOP(index, count) {
			const Dtype t = input[index] + noise[index] * sqrt(1. / (1 + exp(-input[index])));
			output[index] = t * (t > 0);
		}
	}

template <typename Dtype>
void EncodeMachineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// share parameters
	vector<shared_ptr<Blob<Dtype> > > net_params = net_->params();
	for (int i = 0; i < net_params.size(); ++i)
	{
		net_params[i]->ShareData(*blobs_[i]);
		net_params[i]->ShareDiff(*blobs_[i]);
	}

	// copy bottom[0] -> v0_, bottom[0] -> vis_blob_
	caffe_copy(count_v_, bottom[0]->gpu_data(), v0_->mutable_gpu_data());
	caffe_copy(count_v_, bottom[0]->gpu_data(), vis_blob_->mutable_gpu_data());

	// init encode, h0_ -> top[0]
	net_->ForwardFromTo(encode_begin_, encode_end_);
	caffe_copy(count_h_, hid_blob_->gpu_data(), top[0]->mutable_gpu_data());
	caffe_copy(count_h_, hid_blob_->gpu_data(), mean_h0_->mutable_gpu_data());

	// sample hid_blob_
	caffe_gpu_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_gpu_data());
	SampleAct<Dtype> << <CAFFE_GET_BLOCKS(count_h_), CAFFE_CUDA_NUM_THREADS >> >(
		count_h_, hid_blob_->gpu_data(), sample_h_->gpu_data(), hid_blob_->mutable_gpu_data());

	// 1~cd_k decode/encode
	for (int k = 0; k < cd_k_; k++)
	{
		// decode hk-1 -> vk
		net_->ForwardFromTo(decode_begin_, decode_end_);
		// sample vk -> vk
		caffe_gpu_rng_gaussian<Dtype>(count_v_, 0, 1, sample_v_->mutable_gpu_data());
		SampleAct<Dtype> << <CAFFE_GET_BLOCKS(count_v_), CAFFE_CUDA_NUM_THREADS >> >(
			count_v_, re_vis_blob_->gpu_data(), sample_v_->gpu_data(), re_vis_blob_->mutable_gpu_data());

		if (k == 0)
		{
			// calculate loss
			caffe_gpu_sub(count_v_, v0_->gpu_data(), vis_blob_->gpu_data(), diff_v_->mutable_gpu_data());
			Dtype dot;
			caffe_gpu_dot(count_v_, diff_v_->gpu_data(), diff_v_->gpu_data(), &dot);
			Dtype loss = loss_weight_ * dot / bottom[0]->shape(0) / Dtype(2);
			top[1]->mutable_cpu_data()[0] = loss;
		}

		// encode vk -> hk
		caffe_copy(count_v_, re_vis_blob_->gpu_data(), vis_blob_->mutable_gpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);
		// sample hk -> hk
		caffe_gpu_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_gpu_data());
		SampleAct<Dtype> << <CAFFE_GET_BLOCKS(count_h_), CAFFE_CUDA_NUM_THREADS >> >(
			count_h_, hid_blob_->gpu_data(), sample_h_->gpu_data(), hid_blob_->mutable_gpu_data());
	}
	caffe_copy(count_h_, hid_blob_->gpu_data(), hk_->mutable_gpu_data());
	caffe_copy(count_v_, vis_blob_->gpu_data(), vk_->mutable_gpu_data());

}

template <typename Dtype>
void EncodeMachineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
		caffe_copy(count_v_, bottom[0]->gpu_data(), vis_blob_->mutable_gpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_copy(count_h_, top[0]->gpu_diff(), hid_blob_->mutable_gpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);

		// E<p(h, v)> 1
		caffe_copy(count_h_, hk_->gpu_data(), hid_blob_->mutable_gpu_data());
		net_->ForwardFromTo(decode_begin_, decode_end_);

		caffe_gpu_set(count_v_, Dtype(0.), re_vis_blob_->mutable_gpu_diff());
		caffe_gpu_axpy(count_v_, loss_weight_, vk_->gpu_data(), re_vis_blob_->mutable_gpu_diff());
		net_->BackwardFromTo(decode_end_, decode_begin_);

		// E<p(h, v)> 2
		caffe_copy(count_v_, vk_->gpu_data(), vis_blob_->mutable_gpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_gpu_set(count_h_, Dtype(0.), hid_blob_->mutable_gpu_diff());
		caffe_gpu_axpy(count_h_, loss_weight_, hk_->gpu_data(), hid_blob_->mutable_gpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);
	}

	if (s_k_ == 0)
	{
		// sample h0
		caffe_gpu_set(count_h_, Dtype(0.), sample_h_->mutable_gpu_data());
		SampleAct<Dtype> << <CAFFE_GET_BLOCKS(count_h_), CAFFE_CUDA_NUM_THREADS >> >(
			count_h_, mean_h0_->gpu_data(), sample_h_->gpu_data(), h0_->mutable_gpu_data());

		// E<p(h|v)> 1
		caffe_copy(count_h_, h0_->gpu_data(), hid_blob_->mutable_gpu_data());
		net_->ForwardFromTo(decode_begin_, decode_end_);

		caffe_gpu_set(count_v_, Dtype(0.), re_vis_blob_->mutable_gpu_diff());
		caffe_gpu_axpy(count_v_, -loss_weight_, v0_->gpu_data(), re_vis_blob_->mutable_gpu_diff());
		net_->BackwardFromTo(decode_end_, decode_begin_);
		// E<p(h|v)> 2
		caffe_copy(count_v_, v0_->gpu_data(), vis_blob_->mutable_gpu_data());
		net_->ForwardFromTo(encode_begin_, encode_end_);

		caffe_gpu_set(count_h_, Dtype(0.), hid_blob_->mutable_gpu_diff());
		caffe_gpu_axpy(count_h_, -loss_weight_, h0_->gpu_data(), hid_blob_->mutable_gpu_diff());
		net_->BackwardFromTo(encode_end_, encode_begin_);
	}
	else
	{
		for (int k = 0; k < s_k_; ++k)
		{
			// sample h0
			caffe_gpu_rng_gaussian<Dtype>(count_h_, 0, 1, sample_h_->mutable_gpu_data());
			SampleAct<Dtype> << <CAFFE_GET_BLOCKS(count_h_), CAFFE_CUDA_NUM_THREADS >> >(
				count_h_, mean_h0_->gpu_data(), sample_h_->gpu_data(), h0_->mutable_gpu_data());

			// E<p(h|v)> 1
			caffe_copy(count_h_, h0_->gpu_data(), hid_blob_->mutable_gpu_data());
			net_->ForwardFromTo(decode_begin_, decode_end_);

			caffe_gpu_set(count_v_, Dtype(0.), re_vis_blob_->mutable_gpu_diff());
			caffe_gpu_axpy(count_v_, -loss_weight_ / s_k_, v0_->gpu_data(), re_vis_blob_->mutable_gpu_diff());
			net_->BackwardFromTo(decode_end_, decode_begin_);
			// E<p(h|v)> 2
			caffe_copy(count_v_, v0_->gpu_data(), vis_blob_->mutable_gpu_data());
			net_->ForwardFromTo(encode_begin_, encode_end_);

			caffe_gpu_set(count_h_, Dtype(0.), hid_blob_->mutable_gpu_diff());
			caffe_gpu_axpy(count_h_, -loss_weight_ / s_k_, h0_->gpu_data(), hid_blob_->mutable_gpu_diff());
			net_->BackwardFromTo(encode_end_, encode_begin_);
		}
	}

	CUDA_POST_KERNEL_CHECK;
}
INSTANTIATE_LAYER_GPU_FUNCS(EncodeMachineLayer);

}  // namespace caffe
