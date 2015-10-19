
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/10/19
** desc： MapLSTMLayer layer
*********************************************************************************/

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void MapLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);

		const int outer = bottom[0]->count(0, 2);
		const int length = bottom[0]->shape(2);
		const int inner = bottom[0]->count(3);

		conv_->blobs()[0]->ShareData(*(blobs_[0]));
		if (bias_term_)
		{
			conv_->blobs()[1]->ShareData(*(blobs_[1]));
		}
		// 1. copy bottom to X_.
		const Dtype* bottom_data = bottom[0]->gpu_data();
		for (int o = 0; o < outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				Dtype* X_data = X_[l]->mutable_gpu_data();
				caffe_copy(inner, bottom_data, X_data + o * inner);
				bottom_data += inner;
			}
		}
		// For all sequence run lstm.
		for (int t = 0; t < T_; t++)
		{
			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom(2, NULL);
			concat_bottom[0] = X_[t].get();
			if (t == 0)
			{
				concat_bottom[1] = zero_memory_.get();
			}
			else
			{
				concat_bottom[1] = H_1_[t - 1].get();
			}

			const vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
			concat_->Forward(concat_bottom, concat_top);

			//3. forward gate.
			const vector<Blob<Dtype>*> conv_bottom(1, XH_[t].get());
			const vector<Blob<Dtype>*> conv_top(1, G_[t].get());
			conv_->Forward(conv_bottom, conv_top);

			//4. LSTM Unit.
			vector<Blob<Dtype>*> lstm_bottom(2, NULL);
			if (t == 0)
			{
				lstm_bottom[0] = zero_memory_.get();
			}
			else
			{
				lstm_bottom[0] = C_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();

			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_[t].get()
			};
			lstm_unit_->Forward(lstm_bottom, lstm_top);
			// 5 split
			const vector<Blob<Dtype>*> split_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> split_top{ H_1_[t].get(), H_2_[t].get() };
			split_h_->Forward(split_bottom, split_top);
		}
		//6. copy top.
		Dtype* top_data = top[0]->mutable_gpu_data();
		const int out_outer = top[0]->count(0, 2);
		for (int o = 0; o < out_outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				const Dtype* X_data = H_2_[l]->gpu_data();
				caffe_copy(inner, X_data + o * inner, top_data);
				top_data += inner;
			}
		}
	}

	template <typename Dtype>
	void MapLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if (!propagate_down[0])
		{
			return;
		}

		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);

		const int outer = bottom[0]->count(0, 2);
		const int out_outer = top[0]->count(0, 2);
		const int length = bottom[0]->shape(2);
		const int inner = bottom[0]->count(3);;

		conv_->blobs()[0]->ShareData(*(blobs_[0]));
		if (bias_term_)
		{
			conv_->blobs()[1]->ShareData(*(blobs_[1]));
		}
		//6. copy top.
		const Dtype* top_data = top[0]->gpu_diff();
		for (int o = 0; o < out_outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				Dtype* X_data = H_2_[l]->mutable_gpu_diff();
				caffe_copy<Dtype>(inner, top_data, X_data + o * inner);
				top_data += inner;
			}
		}

		// For all sequence run lstm.
		for (int t = T_ - 1; t >= 0; t++)
		{
			// 5 split
			const vector<Blob<Dtype>*> split_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> split_top{ H_1_[t].get(), H_2_[t].get() };
			split_h_->Backward(split_top, vector<bool>(1, true), split_bottom);

			//4. LSTM Unit.
			vector<Blob<Dtype>*> lstm_bottom(2, NULL);
			if (t == 0)
			{
				lstm_bottom[0] = zero_memory_.get();
			}
			else
			{
				lstm_bottom[0] = C_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();
			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_[t].get()
			};
			lstm_unit_->Backward(lstm_top, vector<bool>(2, true), lstm_bottom);

			//3. forward gate.
			const vector<Blob<Dtype>*> conv_bottom(1, XH_[t].get());
			const vector<Blob<Dtype>*> conv_top(1, G_[t].get());
			conv_->Backward(conv_top, vector<bool>(1, true), conv_bottom);

			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom(2, NULL);
			concat_bottom[0] = X_[t].get();
			if (t == 0)
			{
				concat_bottom[1] = zero_memory_.get();
			}
			else
			{
				concat_bottom[1] = H_1_[t - 1].get();
			}

			const vector<Blob<Dtype>*> concat_top(1, XH_[t].get());
			concat_->Backward(concat_top, vector<bool>(2, true), concat_bottom);
		}
		//6. copy top.
		Dtype* bottom_data = bottom[0]->mutable_gpu_diff();
		for (int o = 0; o < outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				const Dtype* X_data = X_[l]->gpu_diff();
				caffe_copy(inner, X_data + o * inner, bottom_data);
				bottom_data += inner;
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MapLSTMLayer);
}  // namespace caffe