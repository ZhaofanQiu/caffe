
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/11
** desc： LSTMUnit layer
*********************************************************************************/

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__device__ Dtype sigmoid(const Dtype x) {
		return Dtype(1) / (Dtype(1) + exp(-x));
	}

	/*
	template <typename Dtype>
	__device__ Dtype tanh(const Dtype x) {
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}
	*/

	template <typename Dtype>
	__global__ void LSTMActsForward(const int nthreads, const int dim,
		const Dtype* X, Dtype* X_acts) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int x_dim = 4 * dim;
			const int d = index % x_dim;
			if (d < 3 * dim) {
				X_acts[index] = sigmoid(X[index]);
			}
			else {
				X_acts[index] = tanh(X[index]);
			}
		}
	}

	template <typename Dtype>
	__global__ void LSTMUnitForward(const int nthreads, const int dim,
		const Dtype* C_prev, const Dtype* X, 
		Dtype* C, Dtype* H) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / dim;
			const int d = index % dim;
			const Dtype* X_offset = X + 4 * dim * n;
			const Dtype i = X_offset[d];
			const Dtype f = X_offset[1 * dim + d];
			const Dtype o = X_offset[2 * dim + d];
			const Dtype g = X_offset[3 * dim + d];
			const Dtype c_prev = C_prev[index];
			const Dtype c = f * c_prev + i * g;
			C[index] = c;
			const Dtype tanh_c = tanh(c);
			H[index] = o * tanh_c;
		}
	}

	template <typename Dtype>
	void LSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int count = top[1]->count();
		const int X_count = bottom[1]->count();
		for (int i = 0; i < this->input_num_; i++)
		{
			const Dtype* C_prev = bottom[i * 2]->gpu_data();
			const Dtype* X = bottom[i * 2 + 1]->gpu_data();
			Dtype* C = top[i * 2]->mutable_gpu_data();
			Dtype* H = top[i * 2 + 1]->mutable_gpu_data();
			Dtype* X_acts = X_acts_[i]->mutable_gpu_data();
			// NOLINT_NEXT_LINE(whitespace/operators)
			LSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS >> >(
				X_count, hidden_dim_, X, X_acts);
			CUDA_POST_KERNEL_CHECK;
			// NOLINT_NEXT_LINE(whitespace/operators)
			LSTMUnitForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, hidden_dim_, C_prev, X_acts, C, H);
			CUDA_POST_KERNEL_CHECK;
		}
	}

	template <typename Dtype>
	__global__ void LSTMUnitBackward(const int nthreads, const int dim,
		const Dtype* C_prev, const Dtype* X, const Dtype* C, const Dtype* H,
		const Dtype* C_diff, const Dtype* H_diff,
		Dtype* C_prev_diff, Dtype* X_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int n = index / dim;
			const int d = index % dim;
			const Dtype* X_offset = X + 4 * dim * n;
			const Dtype i = X_offset[d];
			const Dtype f = X_offset[1 * dim + d];
			const Dtype o = X_offset[2 * dim + d];
			const Dtype g = X_offset[3 * dim + d];
			const Dtype c_prev = C_prev[index];
			const Dtype c = C[index];
			const Dtype tanh_c = tanh(c);
			Dtype* c_prev_diff = C_prev_diff + index;
			Dtype* X_diff_offset = X_diff + 4 * dim * n;
			Dtype* i_diff = X_diff_offset + d;
			Dtype* f_diff = X_diff_offset + 1 * dim + d;
			Dtype* o_diff = X_diff_offset + 2 * dim + d;
			Dtype* g_diff = X_diff_offset + 3 * dim + d;
			const Dtype c_term_diff =
				C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
			*c_prev_diff = c_term_diff * f;
			*i_diff = c_term_diff * g;
			*f_diff = c_term_diff * c_prev;
			*o_diff = H_diff[index] * tanh_c;
			*g_diff = c_term_diff * i;
		}
	}

	template <typename Dtype>
	__global__ void LSTMActsBackward(const int nthreads, const int dim,
		const Dtype* X_acts, const Dtype* X_acts_diff, Dtype* X_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			const int x_dim = 4 * dim;
			const int d = index % x_dim;
			const Dtype X_act = X_acts[index];
			if (d < 3 * dim) {
				X_diff[index] = X_acts_diff[index] * X_act * (Dtype(1) - X_act);
			}
			else {
				X_diff[index] = X_acts_diff[index] * (Dtype(1) - X_act * X_act);
			}
		}
	}

	template <typename Dtype>
	void LSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		bool prop_down = false;
		for (int i = 0; i < propagate_down.size(); i++)
		{
			if (propagate_down[i])
			{
				prop_down = true;
			}
		}
		if (!prop_down)
		{
			return;
		}

		const int count = top[1]->count();
		const int X_count = bottom[1]->count();
		for (int i = 0; i < this->input_num_; i++)
		{
			const Dtype* C_prev = bottom[i * 2]->gpu_data();
			const Dtype* X = bottom[i * 2 + 1]->gpu_data();
			const Dtype* C = top[i * 2]->gpu_data();
			const Dtype* H = top[i * 2 + 1]->gpu_data();
			const Dtype* C_diff = top[i * 2]->gpu_diff();
			const Dtype* H_diff = top[i * 2 + 1]->gpu_diff();
			Dtype* C_prev_diff = bottom[i * 2]->mutable_gpu_diff();
			Dtype* X_diff = bottom[i * 2 + 1]->mutable_gpu_diff();

			Dtype* X_acts = X_acts_[i]->mutable_gpu_data();
			Dtype* X_acts_diff = X_acts_[i]->mutable_gpu_diff();

			LSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS >> >(
				X_count, hidden_dim_, X, X_acts);
			CUDA_POST_KERNEL_CHECK;
			LSTMUnitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
				<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(count, hidden_dim_,
				C_prev, X_acts, C, H, C_diff, H_diff, C_prev_diff, X_acts_diff);
			CUDA_POST_KERNEL_CHECK;
			LSTMActsBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
				<< <CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS >> >(
				X_count, hidden_dim_, X_acts, X_acts_diff, X_diff);
			CUDA_POST_KERNEL_CHECK;
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(LSTMUnitLayer);

}  // namespace caffe