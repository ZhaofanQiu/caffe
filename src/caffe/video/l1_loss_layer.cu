
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： L1Loss layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/l1_loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	void L1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->gpu_data();

		Dtype loss;
		caffe_gpu_asum(count, bottom_data, &loss);
		loss /= bottom[0]->shape(0);
		top[0]->mutable_cpu_data()[0] = loss;
	}

	template <typename Dtype>
	__global__ void L1BackKernel(const int nthreads,
		const Dtype* X, const Dtype eps, Dtype* Y) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			Y[index] = (X[index] > eps) - (X[index] < -eps);
		}
	}

	template <typename Dtype>
	void L1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const int count = bottom[0]->count();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

		L1BackKernel<Dtype> << <CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, this->eps_, bottom_diff);
		caffe_gpu_scal(count, top[0]->cpu_diff()[0], bottom_diff);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(L1LossLayer);

}  // namespace caffe
