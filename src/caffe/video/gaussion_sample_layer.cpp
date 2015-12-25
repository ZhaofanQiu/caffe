
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth£º Zhaofan Qiu
** mail£º zhaofanqiu@gmail.com
** date£º 2015/12/20
** desc£º GaussionSample layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/gaussion_sample_layer.hpp"

namespace caffe {

template <typename Dtype>
void GaussionSampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	N_ = bottom[0]->shape(1);
	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, N_)));
	caffe_set(N_, Dtype(1.), this->blobs_[0]->mutable_cpu_data());
	rng_.reset(new Blob<Dtype>());
	eps_ = 1e-4;
}

template <typename Dtype>
void GaussionSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*bottom[0]);
	rng_->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GaussionSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* sigma_data = this->blobs_[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* rng_data = rng_->mutable_cpu_data();
  int outer = bottom[0]->shape(0);
  int inner = bottom[0]->count(2);

  for (int c = 0; c < N_; ++c)
  {
	  if (sigma_data[c] < eps_)
	  {
		  LOG(INFO) << sigma_data[c] << " < " << eps_;
		  LOG(INFO) << "Force varified.";
		  sigma_data[c] = eps_;
	  }
  }
  if (this->phase_ == TRAIN)
  {
	  caffe::caffe_rng_gaussian(outer * N_ * inner, (Dtype)0, (Dtype)1, rng_data);
  }
  else
  {
	  caffe_set(outer * N_ * inner, (Dtype)0, rng_data);
  }
  int index = 0;
  for (int o = 0; o < outer; ++o)
  {
	  for (int c = 0; c < N_; ++c)
	  {
		  for (int i = 0; i < inner; ++i)
		  {
			  top_data[index] = bottom_data[index] + rng_data[index] * sigma_data[c];
			  index++;
		  }
	  }
  }
}

template <typename Dtype>
void GaussionSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	const Dtype* sigma_data = this->blobs_[0]->cpu_data();
	Dtype* sigma_diff = this->blobs_[0]->mutable_cpu_diff();
	const Dtype* rng_data = rng_->cpu_data();
	int outer = bottom[0]->shape(0);
	int inner = bottom[0]->count(2);

	if (propagate_down[0])
	{
		caffe_copy(outer * N_ * inner, top_diff, bottom_diff);
		caffe_axpy(outer * N_ * inner, Dtype(1.), bottom_data, bottom_diff);
	}
	if (this->param_propagate_down_[0])
	{
		int index = 0;
		for (int o = 0; o < outer; ++o)
		{
			for (int c = 0; c < N_; ++c)
			{
				for (int i = 0; i < inner; ++i)
				{
					sigma_diff[c] += top_diff[index] * rng_data[index] + sigma_data[c] - 1 / sigma_data[c];
					index++;
				}
			}
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(GaussionSampleLayer);
#endif

INSTANTIATE_CLASS(GaussionSampleLayer);
REGISTER_LAYER_CLASS(GaussionSample);

}  // namespace caffe
