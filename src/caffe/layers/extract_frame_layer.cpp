
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/27
** desc： ExtractFrame layer
*********************************************************************************/

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(5, bottom[0]->num_axes());
		frame_id_ = this->layer_param().extract_frame_param().frame();
		CHECK_GE(0, frame_id_);
		CHECK_LT(frame_id_, bottom[0]->shape(2));
	}

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = vector<int>(4, 0);
		top_shape[0] = bottom[0]->shape(0);
		top_shape[1] = bottom[0]->shape(1);
		top_shape[2] = bottom[0]->shape(3);
		top_shape[3] = bottom[0]->shape(4);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int length = bottom[0]->shape(2);
		int outer = bottom[0]->count(0, 2);
		int inner = bottom[0]->count(3);
		caffe_set(top[0]->count(), static_cast<Dtype>(0), top_data);
		for (int o = 0; o < outer; o++)
		{
			caffe_copy(inner, bottom_data + frame_id_ * inner, top_data);
			bottom_data += inner * length;
			top_data += inner;
		}
	}

	template <typename Dtype>
	void ExtractFrameLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		int length = bottom[0]->shape(2);
		int outer = bottom[0]->count(0, 2);
		int inner = bottom[0]->count(3);;
		if (propagate_down[0]) {
			caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
			for (int o = 0; o < outer; o++)
			{
				caffe_copy(inner, top_diff, bottom_diff + frame_id_ * inner);
				bottom_diff += inner * length;
				top_diff += inner;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(ExtractFrameLayer);
#endif

	INSTANTIATE_CLASS(ExtractFrameLayer);
	REGISTER_LAYER_CLASS(ExtractFrame);
}  // namespace caffe
