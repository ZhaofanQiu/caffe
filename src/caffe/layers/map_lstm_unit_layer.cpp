
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
	inline Dtype sigmoid(Dtype x) {
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype x) {
		return 2. * sigmoid(2. * x) - 1.;
	}

	template <typename Dtype>
	void MapLSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		CHECK_EQ(4, bottom[0]->num_axes());
		CHECK_EQ(4, bottom[0]->num_axes());

		CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
		CHECK_EQ(bottom[0]->shape(1) * 4, bottom[1]->shape(1));
		CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
		CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));

		top[0]->ReshapeLike(*bottom[0]);
		top[1]->ReshapeLike(*bottom[0]);
		X_acts_.reset(new Blob<Dtype>(bottom[1]->shape()));
	}

	template <typename Dtype>
	void MapLSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int outer = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		const int inner = bottom[0]->count(1);

		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		Dtype* C = top[0]->mutable_cpu_data();
		Dtype* H = top[1]->mutable_cpu_data();
		
		for (int o = 0; o < outer; ++o)
		{
			for (int ii = 0; ii < inner; ++ii) 
			{
				const Dtype i = sigmoid(X[ii]);
				const Dtype f = sigmoid(X[1 * inner + ii]);
				const Dtype o = sigmoid(X[2 * inner + ii]);
				const Dtype g = tanh(X[3 * inner + ii]);
				const Dtype c_prev = C_prev[ii];
				const Dtype c = f * c_prev + i * g;
				C[ii] = c;
				const Dtype tanh_c = tanh(c);
				H[ii] = o * tanh_c;
			}
			C_prev += inner;
			X += inner * 4;
			C += inner;
			H += inner;
		}
	}

	template <typename Dtype>
	void MapLSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0] && !propagate_down[1])
		{
			return;
		}

		const int outer = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		const int inner = bottom[0]->count(1);

		const Dtype* C_prev = bottom[0]->cpu_data();
		const Dtype* X = bottom[1]->cpu_data();
		const Dtype* C = top[0]->cpu_data();
		const Dtype* H = top[1]->cpu_data();
		const Dtype* C_diff = top[0]->cpu_diff();
		const Dtype* H_diff = top[1]->cpu_diff();
		Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
		Dtype* X_diff = bottom[1]->mutable_cpu_diff();

		for (int o = 0; o < outer; ++o)
		{
			for (int ii = 0; ii < inner; ++ii) {
				const Dtype i = sigmoid(X[ii]);
				const Dtype f = sigmoid(X[1 * inner + ii]);
				const Dtype o = sigmoid(X[2 * inner + ii]);
				const Dtype g = tanh(X[3 * inner + ii]);
				const Dtype c_prev = C_prev[ii];
				const Dtype c = C[ii];
				const Dtype tanh_c = tanh(c);
				Dtype* c_prev_diff = C_prev_diff + ii;
				Dtype* i_diff = X_diff + ii;
				Dtype* f_diff = X_diff + 1 * inner + ii;
				Dtype* o_diff = X_diff + 2 * inner + ii;
				Dtype* g_diff = X_diff + 3 * inner + ii;
				const Dtype c_term_diff =
					C_diff[ii] + H_diff[ii] * o * (1 - tanh_c * tanh_c);
				*c_prev_diff = c_term_diff * f;
				*i_diff = c_term_diff * g * i * (1 - i);
				*f_diff = c_term_diff * c_prev * f * (1 - f);
				*o_diff = H_diff[ii] * tanh_c * o * (1 - o);
				*g_diff = c_term_diff * i * (1 - g * g);;
			}
			C_prev += inner;
			X += inner * 4;
			C += inner;
			H += inner;
			C_diff += inner;
			H_diff += inner;
			X_diff += inner * 4;
			C_prev_diff += inner;
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MapLSTMUnitLayer);
#endif

	INSTANTIATE_CLASS(MapLSTMUnitLayer);
	REGISTER_LAYER_CLASS(MapLSTMUnit);
}  // namespace caffe