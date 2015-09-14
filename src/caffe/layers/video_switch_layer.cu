
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/14
** desc： Video switch layer
*********************************************************************************/

#include <vector>

#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void video_switch_forward_kernel(const int n, const int num,
		const int channels, const int length, const int inner,
		const Dtype* src, Dtype* dest) {
		CUDA_KERNEL_LOOP(index, n) {
			int l = index % length;
			int c = (index / length) % channels;
			int n = index / length / channels;
			int res_offset = ((n * length + l) * channels + c) * inner;
			int des_offset = ((n * channels + c) * length + l) * inner;
			for (int i = 0; i < inner; ++i) {
				dest[des_offset + i] = src[res_offset + i];
			}
		}
	}

	template <typename Dtype>
	void video_switch_forward_gpu(const Dtype* res, Dtype* des,
		const vector<int> res_shape, const vector<int> des_shape,
		int frame_num, int inner)
	{
		const int outer = des_shape[0] * des_shape[1] * des_shape[2];

		// NOLINT_NEXT_LINE(whitespace/operators)
		video_switch_forward_kernel << <CAFFE_GET_BLOCKS(outer), CAFFE_CUDA_NUM_THREADS >> >(
			outer, des_shape[0], des_shape[1],
			des_shape[2], inner, res, des);
	}

	template <typename Dtype>
	__global__ void video_switch_backward_kernel(const int n, const int num,
		const int channels, const int length, const int inner,
		const Dtype* src, Dtype* dest) {
		CUDA_KERNEL_LOOP(index, n) {
			int l = index % length;
			int c = (index / length) % channels;
			int n = index / length / channels;
			int res_offset = ((n * channels + c) * length + l) * inner;
			int des_offset = ((n * length + l) * channels + c) * inner;
			for (int i = 0; i < inner; ++i) {
				dest[des_offset + i] = src[res_offset + i];
			}
		}
	}

	template <typename Dtype>
	void video_switch_backward_gpu(const Dtype* res, Dtype* des,
		const vector<int> res_shape, const vector<int> des_shape,
		int frame_num, int inner)
	{
		const int outer = res_shape[0] * res_shape[1] * res_shape[2];

		// NOLINT_NEXT_LINE(whitespace/operators)
		video_switch_backward_kernel << <CAFFE_GET_BLOCKS(outer), CAFFE_CUDA_NUM_THREADS >> >(
			outer, res_shape[0], res_shape[1],
			res_shape[2], inner, res, des);
	}

	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();

		if (this->to_video_)
		{
			video_switch_forward_gpu(bottom_data, top_data,
				bottom[0]->shape(), top[0]->shape(), frame_num_, top[0]->count(3));
		}
		else
		{
			video_switch_backward_gpu(bottom_data, top_data,
				bottom[0]->shape(), top[0]->shape(), frame_num_, bottom[0]->count(3));
		}
	}

	template <typename Dtype>
	void VideoSwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		if (propagate_down[0]) {
			if (this->to_video_)
			{
				video_switch_backward_gpu(top_diff, bottom_diff,
					top[0]->shape(), bottom[0]->shape(), frame_num_, top[0]->count(3));
			}
			else
			{
				video_switch_forward_gpu(top_diff, bottom_diff,
					top[0]->shape(), bottom[0]->shape(), frame_num_, bottom[0]->count(3));
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(VideoSwitchLayer);
}  // namespace caffe
