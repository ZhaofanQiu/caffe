
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
	void MapLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(5, bottom[0]->num_axes());

		hidden_dim_ = this->layer_param().convolution_param().num_output();
		bias_term_ = this->layer_param().convolution_param().bias_term();
		if (!bias_term_)
		{
			blobs_.resize(1);
		}
		else
		{
			blobs_.resize(2);
		}

		T_ = bottom[0]->shape(2);
		const vector<int> x_shape {
			bottom[0]->shape(0),
			bottom[0]->shape(1),
			bottom[0]->shape(3),
			bottom[0]->shape(4)
		};
		const vector<int> h_shape{
			bottom[0]->shape(0),
			hidden_dim_,
			bottom[0]->shape(3),
			bottom[0]->shape(4)
		};
		const vector<int> xhc_shape {
			bottom[0]->shape(0),
			bottom[0]->shape(1) + hidden_dim_ * 2,
			bottom[0]->shape(3),
			bottom[0]->shape(4)
		};
		const vector<int> gate_shape{
			bottom[0]->shape(0),
			hidden_dim_ * 4,
			bottom[0]->shape(3),
			bottom[0]->shape(4)
		};

		// setup split_h_ layer
		// Bottom & Top
		H_.resize(T_);
		H_1_.resize(T_);
		H_2_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			H_[t].reset(new Blob<Dtype>(h_shape));
			H_1_[t].reset(new Blob<Dtype>(h_shape));
			H_2_[t].reset(new Blob<Dtype>(h_shape));
		}
		zero_memory_.reset(new Blob<Dtype>(h_shape));
		// Layer
		const vector<Blob<Dtype>*> split_h_bottom(1, H_[0].get());
		const vector<Blob<Dtype>*> split_h_top(2, H_1_[0].get());
		split_h_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_h_->SetUp(split_h_bottom, split_h_top);

		// setup concat_h_ layer
		// Bottom & Top
		X_.resize(T_);
		C_.resize(T_);
		XHC_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			X_[t].reset(new Blob<Dtype>(x_shape));
			C_[t].reset(new Blob<Dtype>(h_shape));
			XHC_[t].reset(new Blob<Dtype>(xhc_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> concat_bottom { X_[0].get(),
			H_[0].get(),
			C_[0].get()
		};
		const vector<Blob<Dtype>*> concat_top(1, XHC_[0].get());
		concat_.reset(new ConcatLayer<Dtype>(LayerParameter()));
		concat_->SetUp(concat_bottom, concat_top);

		//setup conv_ layer
		// Top
		G_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			G_[t].reset(new Blob<Dtype>(gate_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> conv_bottom(1, XHC_[0].get());
		const vector<Blob<Dtype>*> conv_top(1, G_[0].get());
		LayerParameter conv_param;
		conv_param.mutable_convolution_param()->CopyFrom(
			this->layer_param().convolution_param());
		conv_param.mutable_convolution_param()->set_num_output(hidden_dim_ * 4);

		conv_.reset(new ConvolutionLayer<Dtype>(conv_param));
		conv_->SetUp(conv_bottom, conv_top);

		blobs_[0].reset(new Blob<Dtype>(conv_->blobs()[0]->shape()));
		blobs_[0]->ShareData(*(conv_->blobs())[0]);
		blobs_[0]->ShareDiff(*(conv_->blobs())[0]);
		if (bias_term_)
		{
			blobs_[1].reset(new Blob<Dtype>(conv_->blobs()[1]->shape()));
			blobs_[1]->ShareData(*(conv_->blobs())[1]);
			blobs_[1]->ShareDiff(*(conv_->blobs())[1]);
		}
		// setup lstm_unit_h_ layer
		// Bottom

		// Layer
		vector<Blob<Dtype>*> lstm_unit_bottom {
			C_[0].get(),
			G_[0].get()
		};
		vector<Blob<Dtype>*> lstm_unit_top{
			C_[0].get(),
			H_[0].get()
		};
		lstm_unit_.reset(new MapLSTMUnitLayer<Dtype>(LayerParameter()));
		lstm_unit_->SetUp(lstm_unit_bottom, lstm_unit_top);

		// setup split_c_ layer
		// Top
		C_1_.resize(T_);
		C_2_.resize(T_);
		for (int t = 0; t < T_; ++t)
		{
			C_1_[t].reset(new Blob<Dtype>(h_shape));
			C_2_[t].reset(new Blob<Dtype>(h_shape));
		}
		// Layer
		vector<Blob<Dtype>*> split_c_bottom(1, C_[0].get());
		vector<Blob<Dtype>*> split_c_top(2, C_1_[0].get());
		split_c_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_c_->SetUp(split_c_bottom, split_c_top);
	}

	template <typename Dtype>
	void MapLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = bottom[0]->shape();
		top_shape[1] = hidden_dim_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void MapLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
		const Dtype* bottom_data = bottom[0]->cpu_data();
		for (int o = 0; o < outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				Dtype* X_data = X_[l]->mutable_cpu_data();
				caffe_copy(inner, bottom_data, X_data + o * inner);
				bottom_data += inner;
			}
		}
		// For all sequence run lstm.
		for (int t = 0; t < T_; t++)
		{
			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom(3, NULL);
			concat_bottom[0] = X_[t].get();
			if (t == 0)
			{
				concat_bottom[1] = zero_memory_.get();
				concat_bottom[2] = zero_memory_.get();
			}
			else
			{
				concat_bottom[1] = H_1_[t - 1].get();
				concat_bottom[2] = C_1_[t - 1].get();
			}

			const vector<Blob<Dtype>*> concat_top(1, XHC_[t].get());
			concat_->Forward(concat_bottom, concat_top);

			//3. forward gate.
			const vector<Blob<Dtype>*> conv_bottom(1, XHC_[t].get());
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
				lstm_bottom[0] = C_2_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();
			
			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_[t].get()
			};
			lstm_unit_->Forward(lstm_bottom, lstm_top);
			// 5 split
			const vector<Blob<Dtype>*> split_h_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> split_h_top{ H_1_[t].get(), H_2_[t].get() };
			split_h_->Forward(split_h_bottom, split_h_top);
			const vector<Blob<Dtype>*> split_c_bottom(1, C_[t].get());
			const vector<Blob<Dtype>*> split_c_top{ C_1_[t].get(), C_2_[t].get() };
			split_c_->Forward(split_c_bottom, split_c_top);
		}
		//6. copy top.
		Dtype* top_data = top[0]->mutable_cpu_data();
		const int out_outer = top[0]->count(0, 2);
		for (int o = 0; o < out_outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				const Dtype* X_data = H_2_[l]->cpu_data();
				caffe_copy(inner, X_data + o * inner, top_data);
				top_data += inner;
			}
		}
	}

	template <typename Dtype>
	void MapLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);

		const int outer = bottom[0]->count(0, 2);
		const int out_outer = top[0]->count(0, 2);
		const int length = bottom[0]->shape(2);
		const int inner = bottom[0]->count(3);

		conv_->blobs()[0]->ShareData(*(blobs_[0]));
		conv_->blobs()[0]->ShareDiff(*(blobs_[0]));
		if (bias_term_)
		{
			conv_->blobs()[1]->ShareData(*(blobs_[1]));
			conv_->blobs()[1]->ShareDiff(*(blobs_[1]));
		}
		//6. copy top.
		const Dtype* top_data = top[0]->cpu_diff();
		for (int o = 0; o < out_outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				Dtype* X_data = H_2_[l]->mutable_cpu_diff();
				caffe_copy<Dtype>(inner, top_data, X_data + o * inner);
				top_data += inner;
			}
		}

		// For all sequence run lstm.
		for (int t = T_ - 1; t >= 0; t--)
		{
			// 5 split
			const vector<Blob<Dtype>*> split_h_bottom(1, H_[t].get());
			const vector<Blob<Dtype>*> split_h_top{ H_1_[t].get(), H_2_[t].get() };
			split_h_->Backward(split_h_top, vector<bool>(1, true), split_h_bottom);
			const vector<Blob<Dtype>*> split_c_bottom(1, C_[t].get());
			const vector<Blob<Dtype>*> split_c_top{ C_1_[t].get(), C_2_[t].get() };
			split_c_->Backward(split_c_top, vector<bool>(1, true), split_c_bottom);
			//4. LSTM Unit.
			vector<Blob<Dtype>*> lstm_bottom(2, NULL);
			if (t == 0)
			{
				lstm_bottom[0] = zero_memory_.get();
			}
			else
			{
				lstm_bottom[0] = C_2_[t - 1].get();
			}
			lstm_bottom[1] = G_[t].get();
			vector<Blob<Dtype>*> lstm_top{
				C_[t].get(),
				H_[t].get()
			};
			lstm_unit_->Backward(lstm_top, vector<bool>(2, true), lstm_bottom);

			//3. forward gate.
			const vector<Blob<Dtype>*> conv_bottom(1, XHC_[t].get());
			const vector<Blob<Dtype>*> conv_top(1, G_[t].get());
			conv_->Backward(conv_top, vector<bool>(1, true), conv_bottom);

			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom(3, NULL);
			concat_bottom[0] = X_[t].get();
			if (t == 0)
			{
				concat_bottom[1] = zero_memory_.get();
				concat_bottom[2] = zero_memory_.get();
			}
			else
			{
				concat_bottom[1] = H_1_[t - 1].get();
				concat_bottom[2] = C_1_[t - 1].get();
			}

			const vector<Blob<Dtype>*> concat_top(1, XHC_[t].get());
			concat_->Backward(concat_top, vector<bool>(3, true), concat_bottom);
		}
		//6. copy top.
		Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
		for (int o = 0; o < outer; ++o)
		{
			for (int l = 0; l < T_; ++l)
			{
				const Dtype* X_data = X_[l]->cpu_diff();
				caffe_copy(inner, X_data + o * inner, bottom_data);
				bottom_data += inner;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(MapLSTMLayer);
#endif

	INSTANTIATE_CLASS(MapLSTMLayer);
	REGISTER_LAYER_CLASS(MapLSTM);
}  // namespace caffe