
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/10/19
** desc： MapLSTMUnit layer
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

	template <typename Dtype>
	__device__ Dtype tanh(const Dtype x) {
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}

	template <typename Dtype>
	__global__ void MapLSTMActsForward(const int nthreads, const int dim,
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
	__global__ void MapLSTMUnitForward(const int nthreads, const int dim,
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
	void MapLSTMUnitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int outer = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		const int inner = bottom[0]->count(1);

		const Dtype* C_prev = bottom[0]->gpu_data();
		const Dtype* X = bottom[1]->gpu_data();
		Dtype* C = top[0]->mutable_gpu_data();
		Dtype* H = top[1]->mutable_gpu_data();
		Dtype* X_acts = X_acts_->mutable_gpu_data();
		// NOLINT_NEXT_LINE(whitespace/operators)
		MapLSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(outer * inner * 4), CAFFE_CUDA_NUM_THREADS >> >(
			outer * inner * 4, inner * 4, X, X_acts);
		CUDA_POST_KERNEL_CHECK;
		// NOLINT_NEXT_LINE(whitespace/operators)
		MapLSTMUnitForward<Dtype> << <CAFFE_GET_BLOCKS(outer * inner), CAFFE_CUDA_NUM_THREADS >> >(
			outer * inner, inner, C_prev, X_acts, C, H);
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	__global__ void MapLSTMUnitBackward(const int nthreads, const int dim,
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
	__global__ void MapLSTMActsBackward(const int nthreads, const int dim,
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
	void MapLSTMUnitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0] && !propagate_down[1])
		{
			return;
		}

		const int outer = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		const int inner = bottom[0]->count(1);

		const Dtype* C_prev = bottom[0]->gpu_data();
		const Dtype* X = bottom[1]->gpu_data();
		const Dtype* C = top[0]->gpu_data();
		const Dtype* H = top[1]->gpu_data();
		const Dtype* C_diff = top[0]->gpu_diff();
		const Dtype* H_diff = top[1]->gpu_diff();
		Dtype* C_prev_diff = bottom[0]->mutable_gpu_diff();
		Dtype* X_diff = bottom[1]->mutable_gpu_diff();

		Dtype* X_acts = X_acts_->mutable_gpu_data();
		Dtype* X_acts_diff = X_acts_->mutable_gpu_diff();

		MapLSTMActsForward<Dtype> << <CAFFE_GET_BLOCKS(outer * inner * 4), CAFFE_CUDA_NUM_THREADS >> >(
			outer * inner * 4, inner * 4, X, X_acts);
		CUDA_POST_KERNEL_CHECK;
		MapLSTMUnitBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<< <CAFFE_GET_BLOCKS(outer * inner), CAFFE_CUDA_NUM_THREADS >> >(outer * inner, inner,
			C_prev, X_acts, C, H, C_diff, H_diff, C_prev_diff, X_acts_diff);
		CUDA_POST_KERNEL_CHECK;
		MapLSTMActsBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
			<< <CAFFE_GET_BLOCKS(outer * inner * 4), CAFFE_CUDA_NUM_THREADS >> >(
			outer * inner * 4, inner * 4, X_acts, X_acts_diff, X_diff);
		CUDA_POST_KERNEL_CHECK;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MapLSTMUnitLayer);
}  // namespace caffe