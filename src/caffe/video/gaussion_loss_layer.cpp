#
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth£º Zhaofan Qiu
** mail£º zhaofanqiu@gmail.com
** date£º 2015/12/20
** desc£º GaussionLoss layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/gaussion_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void GaussionLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		LossLayer<Dtype>::LayerSetUp(bottom, top);
		N_ = bottom[0]->shape(1);
		this->blobs_.resize(1);
		this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, N_)));
		caffe_set(N_, Dtype(1.), this->blobs_[0]->mutable_cpu_data());
		eps_ = 1e-4;
	}

template <typename Dtype>
void GaussionLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GaussionLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int outer = bottom[0]->shape(0);
  int inner = bottom[0]->count(2);
  Dtype* diff_data = diff_.mutable_cpu_data();
  Dtype* sigma_data = this->blobs_[0]->mutable_cpu_data();

  for (int c = 0; c < N_; ++c)
  {
	  if (sigma_data[c] < eps_)
	  {
		  LOG(INFO) << sigma_data[c] << " < " << eps_;
		  LOG(INFO) << "Force varified.";
		  sigma_data[c] = eps_;
	  }
  }

  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_data);
  Dtype dot = 0;
  int index = 0;
  for (int o = 0; o < outer; ++o)
  {
	  for (int c = 0; c < N_; ++c)
	  {
		  for (int i = 0; i < inner; ++i)
		  {
			  dot += diff_data[index] * diff_data[index] / 2 / sigma_data[c] / sigma_data[c] + log(sigma_data[c]) + log(2 * 3.1415926) * 0.5;
			  index++;
		  }
	  }
  }
  Dtype loss = dot;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void GaussionLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count = bottom[0]->count();
	int outer = bottom[0]->shape(0);
	int inner = bottom[0]->count(2);
	const Dtype* diff_data = diff_.cpu_data();
	const Dtype* sigma_data = this->blobs_[0]->cpu_data();

  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
	  const Dtype alpha = sign * top[0]->cpu_diff()[0];
	  int index = 0;
	  Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
	  for (int o = 0; o < outer; ++o)
	  {
		  for (int c = 0; c < N_; ++c)
		  {
			  for (int i = 0; i < inner; ++i)
			  {
				  bottom_diff[index] = alpha * diff_data[index] / sigma_data[c] / sigma_data[c];
				  index++;
			  }
		  }
	  }
    }
  }
  if (this->param_propagate_down_[0])
  {
	  int index = 0;
	  const Dtype alpha = top[0]->cpu_diff()[0];
	  Dtype* sigma_diff = this->blobs_[0]->mutable_cpu_diff();
	  for (int o = 0; o < outer; ++o)
	  {
		  for (int c = 0; c < N_; ++c)
		  {
			  for (int i = 0; i < inner; ++i)
			  {
				  sigma_diff[c] += alpha * ( -diff_data[index] * diff_data[index] / sigma_data[c] / sigma_data[c] / sigma_data[c]
					  + 1 / sigma_data[c]);
				  index++;
			  }
		  }
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(GaussionLossLayer);
#endif

INSTANTIATE_CLASS(GaussionLossLayer);
REGISTER_LAYER_CLASS(GaussionLoss);

}  // namespace caffe
