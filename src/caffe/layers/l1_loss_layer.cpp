
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/11/30
** desc： L1Loss layer
*********************************************************************************/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

template <typename Dtype>
void L1LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	this->eps_ = 1e-3;
}

template <typename Dtype>
void L1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(vector<int>(0));
}

template <typename Dtype>
void L1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = 0;
  for (int i = 0; i < count; ++i)
  {
	  top_data[0] += abs(bottom_data[i]);
  }
  top_data[0] /= bottom[0]->shape(0);
}

template <typename Dtype>
void L1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < count; ++i)
  {
	  bottom_diff[i] = (bottom_data[i] > eps_) - (bottom_data[i] < -eps_);
  }
  caffe_scal(count, top[0]->cpu_diff()[0], bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(L1LossLayer);
#endif

INSTANTIATE_CLASS(L1LossLayer);
REGISTER_LAYER_CLASS(L1Loss);

}  // namespace caffe
