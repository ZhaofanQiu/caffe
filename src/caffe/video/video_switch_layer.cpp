
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： VideoSwitch layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/video_switch_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		VideoSwitchParameter param = this->layer_param_.video_switch_param();
		CHECK(param.has_switch_())
			<< "Video Switch Layer must have switch";
		this->to_video_ = (param.switch_() == VideoSwitchParameter_SwitchOp_VIDEO);
		if (this->to_video_)
		{
			CHECK(param.has_frame_num())
				<< "Switch to video must have frame_num";
			frame_num_ = param.frame_num();
			CHECK(bottom[0]->shape(0) % frame_num_ == 0);
		}
		else
		{
			frame_num_ = bottom[0]->shape(2);
		}
	}

	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> bottom_shape = bottom[0]->shape();
		vector<int> top_shape;
		if (this->to_video_)
		{
			top_shape = vector<int>(bottom_shape.size() + 1, 0);
			top_shape[0] = bottom_shape[0] / frame_num_;
			top_shape[1] = bottom_shape[1];
			top_shape[2] = frame_num_;
			for (int i = 3; i < top_shape.size(); i++)
			{
				top_shape[i] = bottom_shape[i - 1];
			}
		}
		else
		{
			top_shape = vector<int>(bottom_shape.size() - 1, 0);
			top_shape[0] = bottom_shape[0] * bottom_shape[2];
			top_shape[1] = bottom_shape[1];
			for (int i = 2; i < top_shape.size(); i++)
			{
				top_shape[i] = bottom_shape[i + 1];
			}
		}
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void video_switch_forward_cpu(const Dtype* res, Dtype* des,
		const vector<int> res_shape, const vector<int> des_shape,
		int frame_num, int inner)
	{
		int res_offset = 0;
		int des_offset = 0;
		for (int n = 0; n < des_shape[0]; n++)
		{
			for (int c = 0; c < des_shape[1]; c++)
			{
				for (int l = 0; l < des_shape[2]; l++)
				{
					res_offset = ((n * frame_num + l) * res_shape[1] + c) * inner;

					caffe_copy(inner, res + res_offset, des + des_offset);
					des_offset += inner;
				}
			}
		}
	}

	template <typename Dtype>
	void video_switch_backward_cpu(const Dtype* res, Dtype* des,
		const vector<int> res_shape, const vector<int> des_shape,
		int frame_num, int inner)
	{
		int res_offset = 0;
		int des_offset = 0;
		for (int n = 0; n < res_shape[0]; n++)
		{
			for (int c = 0; c < res_shape[1]; c++)
			{
				for (int l = 0; l < res_shape[2]; l++)
				{
					des_offset = ((n * frame_num + l) * res_shape[1] + c) * inner;

					caffe_copy(inner, res + res_offset, des + des_offset);
					res_offset += inner;
				}
			}
		}
	}
	
	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		if (this->to_video_)
		{
			video_switch_forward_cpu(bottom_data, top_data,
				bottom[0]->shape(), top[0]->shape(), frame_num_, top[0]->count(3));
		}
		else
		{
			video_switch_backward_cpu(bottom_data, top_data,
				bottom[0]->shape(), top[0]->shape(), frame_num_, bottom[0]->count(3));
		}
	}

	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		if (propagate_down[0]) {
			if (this->to_video_)
			{
				video_switch_backward_cpu(top_diff, bottom_diff,
					top[0]->shape(), bottom[0]->shape(), frame_num_, top[0]->count(3));
			}
			else
			{
				video_switch_forward_cpu(top_diff, bottom_diff,
					top[0]->shape(), bottom[0]->shape(), frame_num_, bottom[0]->count(3));
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(VideoSwitchLayer);
#endif

	INSTANTIATE_CLASS(VideoSwitchLayer);
	REGISTER_LAYER_CLASS(VideoSwitch);
}  // namespace caffe
