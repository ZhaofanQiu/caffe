
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/27
** desc： ExtractFrame layer
*********************************************************************************/

#include <vector>

#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		int length = bottom[0]->shape(2);
		int outer = bottom[0]->count(0, 2);
		int inner = bottom[0]->count(3);
		caffe_gpu_set(top[0]->count(), static_cast<Dtype>(0), top_data);
		for (int o = 0; o < outer; o++)
		{
			caffe_copy(inner, bottom_data + frame_id_ * inner, top_data);
			bottom_data += inner * length;
			top_data += inner;
		}
	}

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		int length = bottom[0]->shape(2);
		int outer = bottom[0]->count(0, 2);
		int inner = bottom[0]->count(3);
		if (propagate_down[0]) {
			caffe_gpu_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
			for (int o = 0; o < outer; o++)
			{
				caffe_copy(inner, top_diff, bottom_diff + frame_id_ * inner);
				bottom_diff += inner * length;
				top_diff += inner;
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ExtractFrameLayer);
}  // namespace caffe
