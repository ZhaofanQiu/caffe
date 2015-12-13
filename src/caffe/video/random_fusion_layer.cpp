
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
void RandomFusionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_GT(bottom.size(), top.size());
	if (this->layer_param().random_fusion_param().has_std())
	{
		std_ = this->layer_param().random_fusion_param().std();
	}
	if (this->layer_param().random_fusion_param().has_prob())
	{
		prob_ = this->layer_param().random_fusion_param().prob();
		CHECK(prob_ > 0.);
		CHECK(prob_ < 1.);
	}
	if (this->layer_param().random_fusion_param().has_mean())
	{
		mean_ = this->layer_param().random_fusion_param().mean();
	}
	random_vec_ = vector<Dtype>(bottom.size(), 1);
	random_idx_ = vector<unsigned int>(bottom.size(), 1);
}

template <typename Dtype>
void RandomFusionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	for (int i = 1; i < bottom.size(); ++i) {
		CHECK(bottom[i]->shape() == bottom[0]->shape());
	}
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RandomFusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();

  caffe_set(count, Dtype(0.), top_data);
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
	  caffe_axpy(count, random_vec_[i], bottom[i]->cpu_data(), top_data);
  }
}

template <typename Dtype>
void RandomFusionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	const int count = top[0]->count();
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	for (int i = 0; i < bottom.size(); ++i) {
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
		if (propagate_down[i])
		{
			caffe_cpu_scale(count, random_vec_[i], top_diff, bottom_diff);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(RandomFusionLayer);
#endif

INSTANTIATE_CLASS(RandomFusionLayer);
REGISTER_LAYER_CLASS(RandomFusion);

}  // namespace caffe
