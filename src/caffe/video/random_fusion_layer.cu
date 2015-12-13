
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth£º Zhaofan Qiu
** mail£º zhaofanqiu@gmail.com
** date£º 2015/12/13
** desc£º RandomFusion layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/random_fusion_layer.hpp"

namespace caffe {

template <typename Dtype>
void RandomFusionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = top[0]->count();

	caffe_gpu_set(count, Dtype(0.), top_data);
	if (this->phase_ == TRAIN)
	{
		switch (this->layer_param().random_fusion_param().random())
		{
		case caffe::RandomFusionParameter_RandomMethod_Gaussion:
		  caffe::caffe_rng_gaussian(bottom.size(), (Dtype)mean_, (Dtype)std_, &random_vec_[0]);
		  for (int i = 0; i < bottom.size(); i++)
		  {
			  random_vec_[i] = std::max(random_vec_[i], (Dtype)0.);
		  }
		  break;
		case caffe::RandomFusionParameter_RandomMethod_Bernoulli:
			caffe::caffe_rng_bernoulli(bottom.size(), (Dtype)prob_, &random_idx_[0]);
			for (int i = 0; i < bottom.size(); i++)
			{
				random_vec_[i] = static_cast<Dtype>(random_idx_[i]);
			}
			break;
		default:
			LOG(FATAL) << "Unknown random operation.";
		}
		Dtype sum = static_cast<Dtype>(0.);
		for (int i = 0; i < bottom.size(); i++)
		{
			sum += random_vec_[i];
		}
		if (sum < 1e-6)
		{
			for (int i = 0; i < bottom.size(); i++)
			{
				random_vec_[i] = static_cast<Dtype>(1.0 / bottom.size());
			}
		}
		else
		{
			for (int i = 0; i < bottom.size(); i++)
			{
				random_vec_[i] /= sum;
			}
		}
	}
	else
	{
		for (int i = 0; i < bottom.size(); i++)
		{
			random_vec_[i] = static_cast<Dtype>(1.0 / bottom.size());
		}
	}
	for (int i = 0; i < bottom.size(); ++i) {
		caffe_gpu_axpy(count, random_vec_[i], bottom[i]->gpu_data(), top_data);
	}
}

template <typename Dtype>
void RandomFusionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const int count = top[0]->count();
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	for (int i = 0; i < bottom.size(); ++i) {
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
		if (propagate_down[i])
		{
			caffe_gpu_scale(count, random_vec_[i], top_diff, bottom_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(RandomFusionLayer);


}  // namespace caffe
