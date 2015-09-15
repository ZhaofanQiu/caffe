
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
	inline Dtype sigmoid(Dtype x) {
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype tanh(Dtype x) {
		return 2. * sigmoid(2. * x) - 1.;
	}

	template <typename Dtype>
	void LSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(0, bottom.size() % 2);
		this->input_num_ = bottom.size() / 2;
		this->hidden_dim_ = bottom[0]->shape(1);

		for (int i = 0; i < this->input_num_; ++i) {
			CHECK_EQ(2, bottom[i * 2]->num_axes());
			CHECK_EQ(2, bottom[i * 2 + 1]->num_axes());
			CHECK_EQ(this->hidden_dim_, bottom[i * 2]->shape(1));
			CHECK_EQ(this->hidden_dim_ * 4, bottom[i * 2 + 1]->shape(1));
			CHECK_EQ(bottom[0]->shape(0), bottom[i * 2]->shape(0));
			CHECK_EQ(bottom[0]->shape(0), bottom[i * 2 + 1]->shape(0));
		}

		X_acts_.resize(this->input_num_);
		for (int i = 0; i < this->input_num_; i++)
		{
			top[i * 2]->ReshapeLike(*bottom[0]);
			top[i * 2 + 1]->ReshapeLike(*bottom[0]);
			X_acts_[i].reset(new Blob<Dtype>(bottom[1]->shape()));
		}
	}

	template <typename Dtype>
	void LSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num = bottom[0]->shape(0);
		const int x_dim = hidden_dim_ * 4;

		for (int i = 0; i < this->input_num_; i++)
		{
			const Dtype* C_prev = bottom[i * 2]->cpu_data();
			const Dtype* X = bottom[i * 2 + 1]->cpu_data();
			Dtype* C = top[i * 2]->mutable_cpu_data();
			Dtype* H = top[i * 2 + 1]->mutable_cpu_data();
			for (int n = 0; n < num; ++n) {
				for (int d = 0; d < hidden_dim_; ++d) {
					const Dtype i = sigmoid(X[d]);
					const Dtype f =	sigmoid(X[1 * hidden_dim_ + d]);
					const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
					const Dtype g = tanh(X[3 * hidden_dim_ + d]);
					const Dtype c_prev = C_prev[d];
					const Dtype c = f * c_prev + i * g;
					C[d] = c;
					const Dtype tanh_c = tanh(c);
					H[d] = o * tanh_c;
				}
				C_prev += hidden_dim_;
				X += x_dim;
				C += hidden_dim_;
				H += hidden_dim_;
			}
		}
	}

	template <typename Dtype>
	void LSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
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

		const int num = bottom[0]->shape(0);
		const int x_dim = hidden_dim_ * 4;
		for (int i = 0; i < this->input_num_; i++)
		{
			const Dtype* C_prev = bottom[i * 2]->cpu_data();
			const Dtype* X = bottom[i * 2 + 1]->cpu_data();
			const Dtype* C = top[i * 2]->cpu_data();
			const Dtype* H = top[i * 2 + 1]->cpu_data();

			const Dtype* C_diff = top[i * 2]->cpu_diff();
			const Dtype* H_diff = top[i * 2 + 1]->cpu_diff();
			Dtype* C_prev_diff = bottom[i * 2]->mutable_cpu_diff();
			Dtype* X_diff = bottom[i * 2 + 1]->mutable_cpu_diff();
			for (int n = 0; n < num; ++n) {
				for (int d = 0; d < hidden_dim_; ++d) {
					const Dtype i = sigmoid(X[d]);
					const Dtype f =	sigmoid(X[1 * hidden_dim_ + d]);
					const Dtype o = sigmoid(X[2 * hidden_dim_ + d]);
					const Dtype g = tanh(X[3 * hidden_dim_ + d]);
					const Dtype c_prev = C_prev[d];
					const Dtype c = C[d];
					const Dtype tanh_c = tanh(c);
					Dtype* c_prev_diff = C_prev_diff + d;
					Dtype* i_diff = X_diff + d;
					Dtype* f_diff = X_diff + 1 * hidden_dim_ + d;
					Dtype* o_diff = X_diff + 2 * hidden_dim_ + d;
					Dtype* g_diff = X_diff + 3 * hidden_dim_ + d;
					const Dtype c_term_diff =
						C_diff[d] + H_diff[d] * o * (1 - tanh_c * tanh_c);
					*c_prev_diff = c_term_diff * f;
					*i_diff = c_term_diff * g * i * (1 - i);
					*f_diff = c_term_diff * c_prev * f * (1 - f);
					*o_diff = H_diff[d] * tanh_c * o * (1 - o);
					*g_diff = c_term_diff * i * (1 - g * g);
				}
				C_prev += hidden_dim_;
				X += x_dim;
				C += hidden_dim_;
				H += hidden_dim_;
				C_diff += hidden_dim_;
				H_diff += hidden_dim_;
				X_diff += x_dim;
				C_prev_diff += hidden_dim_;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(LSTMUnitLayer);
#endif

	INSTANTIATE_CLASS(LSTMUnitLayer);
	REGISTER_LAYER_CLASS(LSTMUnit);
}  // namespace caffe