
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
void GaussionSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void GaussionSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(GaussionSampleLayer);


}  // namespace caffe
