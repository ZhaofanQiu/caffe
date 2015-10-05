
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/15
** desc： GirdLSTMLayer layer
*********************************************************************************/

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		CHECK_LE(3, bottom[0]->num_axes());
		grid_dim_ = bottom[0]->num_axes() - 2;
		reverse_ = vector<bool>(grid_dim_, false);
		CHECK_GE(grid_dim_, reverse_.size());
		CHECK(this->layer_param().inner_product_param().has_num_output());
		for (int i = 0; i < this->layer_param().grid_lstm_param().reverse_size(); i++)
		{
			reverse_[i] = this->layer_param().grid_lstm_param().reverse(i);
		}
		num_seq_ = bottom[0]->count(2);
		hidden_dim_ = this->layer_param().inner_product_param().num_output();
		bias_term_ = this->layer_param().inner_product_param().bias_term();
		if (!bias_term_)
		{
			blobs_.resize(grid_dim_);
		}
		else
		{
			blobs_.resize(grid_dim_ * 2);
		}
		this->CalculateOrder(bottom);

		const int batch_size = bottom[0]->shape(0);
		const int x_dim = bottom[0]->shape(1);
		output_dim_ = hidden_dim_ * grid_dim_ + x_dim;
		int top_blob = 0;
		// setup split_x_ layer
		// Bottom & Top
		const vector<int> x_shape{ batch_size, x_dim };
		X_.resize(num_seq_);
		X_1_.resize(num_seq_);
		X_2_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			X_[i].reset(new Blob<Dtype>(x_shape));
			X_1_[i].reset(new Blob<Dtype>(x_shape));
			X_2_[i].reset(new Blob<Dtype>(x_shape));
		}
		// Layer
		const vector<Blob<Dtype>*> split_x_bottom(1, X_[0].get());
		const vector<Blob<Dtype>*> split_x_top(2, X_1_[0].get());
		split_x_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_x_->SetUp(split_x_bottom, split_x_top);

		// setup split_h_ layer
		// Bottom & Top
		const vector<int> h_i_shape{ batch_size, hidden_dim_ };
		H_i_.resize(num_seq_);
		H_i_1_.resize(num_seq_);
		H_i_2_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			H_i_[i].resize(grid_dim_);
			H_i_1_[i].resize(grid_dim_);
			H_i_2_[i].resize(grid_dim_);
			for (int j = 0; j < grid_dim_; ++j)
			{
				H_i_[i][j].reset(new Blob<Dtype>(h_i_shape));
				H_i_1_[i][j].reset(new Blob<Dtype>(h_i_shape));
				H_i_2_[i][j].reset(new Blob<Dtype>(h_i_shape));
			}
		}
		zero_memory_.reset(new Blob<Dtype>(h_i_shape));
		// Layer
		const vector<Blob<Dtype>*> split_h_bottom(1, H_i_[0][0].get());
		const vector<Blob<Dtype>*> split_h_top(2, H_i_1_[0][0].get());
		split_h_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_h_->SetUp(split_h_bottom, split_h_top);

		// setup split_xh_h_ layer
		// Bottom & Top
		const vector<int> xh_h_shape{ batch_size, hidden_dim_ * grid_dim_ + x_dim };
		XH_h_.resize(num_seq_);
		XH_h_k_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			XH_h_[i].reset(new Blob<Dtype>(xh_h_shape));
			XH_h_k_[i].resize(grid_dim_);
			for (int j = 0; j < grid_dim_; j++)
			{
				XH_h_k_[i][j].reset(new Blob<Dtype>(xh_h_shape));
			}
		}
		// Layer
		const vector<Blob<Dtype>*> split_xh_h_bottom(1, XH_h_[0].get());
		const vector<Blob<Dtype>*> split_xh_h_top(grid_dim_, XH_h_k_[0][0].get());
		split_xh_h_.reset(new SplitLayer<Dtype>(LayerParameter()));
		split_xh_h_->SetUp(split_xh_h_bottom, split_xh_h_top);

		// setup concat_h_ layer
		// Layer
		vector<Blob<Dtype>*> concat_h_bottom(1 + grid_dim_, H_i_[0][0].get());
		concat_h_bottom[0] = X_[0].get();
		const vector<Blob<Dtype>*> concat_h_top(1, XH_h_[0].get());
		concat_h_.reset(new ConcatLayer<Dtype>(LayerParameter()));
		concat_h_->SetUp(concat_h_bottom, concat_h_top);

		// setup concat_x_ layer
		// Top
		const vector<int> xh_x_shape{ batch_size, hidden_dim_ * grid_dim_ + x_dim };
		XH_x_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			XH_x_[i].reset(new Blob<Dtype>(xh_x_shape));
		}
		// Layer
		vector<Blob<Dtype>*> concat_x_bottom(1 + grid_dim_, H_i_[0][0].get());
		concat_x_bottom[0] = X_[0].get();
		const vector<Blob<Dtype>*> concat_x_top(1, XH_x_[0].get());
		concat_x_.reset(new ConcatLayer<Dtype>(LayerParameter()));
		concat_x_->SetUp(concat_x_bottom, concat_x_top);

		//setup ip_xh_h_ layer
		// Top
		const vector<int> g_h_shape{ batch_size, hidden_dim_ * 4 };
		G_h_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			G_h_[i].resize(grid_dim_);
			for (int j = 0; j < grid_dim_; ++j)
			{
				G_h_[i][j].reset(new Blob<Dtype>(g_h_shape));
			}
		}
		// Layer
		const vector<Blob<Dtype>*> ip_xh_h_bottom(1, XH_h_[0].get());
		const vector<Blob<Dtype>*> ip_xh_h_top(1, G_h_[0][0].get());
		LayerParameter ip_xh_h_param;
		ip_xh_h_param.mutable_inner_product_param()->CopyFrom(
			this->layer_param().inner_product_param());
		ip_xh_h_param.mutable_inner_product_param()->set_num_output(hidden_dim_ * 4);
		ip_xh_h_.resize(grid_dim_);
		for (int i = 0; i < grid_dim_; ++i)
		{
			ip_xh_h_[i].reset(new InnerProductLayer<Dtype>(ip_xh_h_param));
			ip_xh_h_[i]->SetUp(ip_xh_h_bottom, ip_xh_h_top);
			blobs_[top_blob].reset(new Blob<Dtype>(ip_xh_h_[i]->blobs()[0]->shape()));
			blobs_[top_blob]->ShareData(*(ip_xh_h_[i]->blobs())[0]);
			blobs_[top_blob++]->ShareDiff(*(ip_xh_h_[i]->blobs())[0]);
			if (bias_term_)
			{
				blobs_[top_blob].reset(new Blob<Dtype>(ip_xh_h_[i]->blobs()[1]->shape()));
				blobs_[top_blob]->ShareData(*(ip_xh_h_[i]->blobs())[1]);
				blobs_[top_blob++]->ShareDiff(*(ip_xh_h_[i]->blobs())[1]);
			}
		}

		// setup lstm_unit_h_ layer
		// Bottom
		const vector<int> c_i_shape{ batch_size, hidden_dim_ };
		C_i_.resize(num_seq_);
		for (int i = 0; i < num_seq_; ++i)
		{
			C_i_[i].resize(grid_dim_);
			for (int j = 0; j < grid_dim_; ++j)
			{
				C_i_[i][j].reset(new Blob<Dtype>(c_i_shape));
			}
		}
		// Layer
		vector<Blob<Dtype>*> lstm_unit_h_bottom_vec(grid_dim_ * 2, NULL);
		for (int i = 0; i < grid_dim_; i++)
		{
			lstm_unit_h_bottom_vec[i * 2] = C_i_[0][0].get();
			lstm_unit_h_bottom_vec[i * 2 + 1] = G_h_[0][0].get();
		}
		vector<Blob<Dtype>*> lstm_unit_h_top_vec(grid_dim_ * 2, NULL);
		for (int i = 0; i < grid_dim_; i++)
		{
			lstm_unit_h_top_vec[i * 2] = C_i_[0][0].get();
			lstm_unit_h_top_vec[i * 2 + 1] = H_i_[0][0].get();
		}
		lstm_unit_h_.reset(new LSTMUnitLayer<Dtype>(LayerParameter()));
		lstm_unit_h_->SetUp(lstm_unit_h_bottom_vec, lstm_unit_h_top_vec);
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = bottom[0]->shape();
		top_shape[1] = output_dim_;
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		for (int i = 0; i < grid_dim_; i++)
		{
			if (bias_term_)
			{
				(ip_xh_h_[i]->blobs()[0])->ShareData(*(blobs_[i * 2]));
				(ip_xh_h_[i]->blobs()[1])->ShareData(*(blobs_[i * 2 + 1]));
			}
			else
			{
				(ip_xh_h_[i]->blobs()[0])->ShareData(*(blobs_[i]));
			}
		}
		// 1. copy bottom to X_.
		int idx = 0;
		const Dtype* bottom_data = bottom[0]->cpu_data();
		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < channels; c++)
			{
				for (int d = 0; d < num_seq_; d++)
				{
					Dtype* X_data = X_[d]->mutable_cpu_data();
					X_data[idx] = *(bottom_data++);
				}
				idx++;
			}
		}
		// 1.5 split
		for (int i = 0; i < num_seq_; i++)
		{
			const vector<Blob<Dtype>*> split2_bottom_vec(1, X_[i].get());
			const vector<Blob<Dtype>*> split2_top_vec{ X_1_[i].get(), X_2_[i].get() };
			split_x_->Forward(split2_bottom_vec, split2_top_vec);
		}
		// For all sequence run lstm.
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[d];
			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom_vec(grid_dim_ + 1, NULL);
			concat_bottom_vec[0] = X_1_[dp].get();
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					concat_bottom_vec[1 + i] = zero_memory_.get();
				}
				else
				{
					concat_bottom_vec[1 + i] = H_i_1_[link_idx_[dp][i]][i].get();
				}
			}
			const vector<Blob<Dtype>*> concat_top_vec(1, XH_h_[dp].get());
			concat_h_->Forward(concat_bottom_vec, concat_top_vec);
			//2.5. split xh_h.
			const vector<Blob<Dtype>*> split_xh_bottom_vec(1, XH_h_[dp].get());
			vector<Blob<Dtype>*> split_xh_top_vec(grid_dim_, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				split_xh_top_vec[i] = XH_h_k_[dp][i].get();
			}
			split_xh_h_->Forward(split_xh_bottom_vec, split_xh_top_vec);
			//3. forward gate.
			for (int i = 0; i < grid_dim_; i++)
			{
				const vector<Blob<Dtype>*> ip_bottom_vec(1, XH_h_k_[dp][i].get());
				const vector<Blob<Dtype>*> ip_top_vec(1, G_h_[dp][i].get());
				ip_xh_h_[i]->Forward(ip_bottom_vec, ip_top_vec);
			}
			//4. LSTM Unit 1.
			vector<Blob<Dtype>*> lstm_bottom_vec(grid_dim_ * 2, NULL);
			vector<Blob<Dtype>*> lstm_top_vec(grid_dim_ * 2, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					lstm_bottom_vec[i * 2] = zero_memory_.get();
				}
				else
				{
					lstm_bottom_vec[i * 2] = C_i_[link_idx_[dp][i]][i].get();
				}
				lstm_bottom_vec[i * 2 + 1] = G_h_[dp][i].get();
				lstm_top_vec[i * 2] = C_i_[dp][i].get();
				lstm_top_vec[i * 2 + 1] = H_i_[dp][i].get();
			}
			lstm_unit_h_->Forward(lstm_bottom_vec, lstm_top_vec);
			// 4.5 split
			for (int i = 0; i < grid_dim_; i++)
			{
				const vector<Blob<Dtype>*> split1_bottom_vec(1, H_i_[dp][i].get());
				const vector<Blob<Dtype>*> split1_top_vec{ H_i_1_[dp][i].get(), H_i_2_[dp][i].get() };
				split_h_->Forward(split1_bottom_vec, split1_top_vec);
			}
		}
		// Concat output.
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[d];
			//5. concat x & h_t
			vector<Blob<Dtype>*> concat_bottom_vec(1 + grid_dim_, NULL);
			concat_bottom_vec[0] = X_2_[dp].get();
			for (int i = 0; i < grid_dim_; i++)
			{
				concat_bottom_vec[1 + i] = H_i_2_[dp][i].get();
			}
			const vector<Blob<Dtype>*> concat_top_vec(1, XH_x_[dp].get());
			concat_x_->Forward(concat_bottom_vec, concat_top_vec);
		}
		//6. copy top.
		idx = 0;
		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < output_dim_; c++)
			{
				for (int d = 0; d < num_seq_; d++)
				{
					const Dtype* X_data = XH_x_[d]->cpu_data();
					*(top_data++) = X_data[idx];
				}
				idx++;
			}
		}
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		for (int i = 0; i < grid_dim_; i++)
		{
			if (bias_term_)
			{
				(ip_xh_h_[i]->blobs()[0])->ShareData(*(blobs_[i * 2]));
				(ip_xh_h_[i]->blobs()[0])->ShareDiff(*(blobs_[i * 2]));
				(ip_xh_h_[i]->blobs()[1])->ShareData(*(blobs_[i * 2 + 1]));
				(ip_xh_h_[i]->blobs()[1])->ShareDiff(*(blobs_[i * 2 + 1]));
			}
			else
			{
				(ip_xh_h_[i]->blobs()[0])->ShareData(*(blobs_[i]));
				(ip_xh_h_[i]->blobs()[0])->ShareDiff(*(blobs_[i]));
			}
		}
		// Concat output.
		//6. copy top.
		int idx = 0;
		const Dtype* top_data = top[0]->cpu_diff();
		for (int n = 0; n < num; n++)
		{
			for (int c = 0; c < output_dim_; c++)
			{
				for (int d = 0; d < num_seq_; d++)
				{
					Dtype* X_diff = XH_x_[d]->mutable_cpu_diff();
					X_diff[idx] = *(top_data++);
				}
				idx++;
			}
		}
		//5. concat x & h_t
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[num_seq_ - 1 - d];
			vector<Blob<Dtype>*> concat_bottom_vec(1 + grid_dim_, NULL);
			concat_bottom_vec[0] = X_2_[dp].get();
			for (int i = 0; i < grid_dim_; i++)
			{
				concat_bottom_vec[1 + i] = H_i_2_[dp][i].get();
			}
			const vector<Blob<Dtype>*> concat_top_vec(1, XH_x_[dp].get());
			concat_x_->Backward(concat_top_vec, 
				vector<bool>(1 + grid_dim_, true), concat_bottom_vec);
		}

		// For all sequence run lstm.
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[num_seq_ - 1 - d];
			// 4.5 split
			for (int i = 0; i < grid_dim_; i++)
			{
				const vector<Blob<Dtype>*> split1_bottom_vec(1, H_i_[dp][i].get());
				const vector<Blob<Dtype>*> split1_top_vec{ H_i_1_[dp][i].get(), H_i_2_[dp][i].get() };
				split_h_->Backward(split1_top_vec,
					vector<bool>(1, true), split1_bottom_vec);
			}
			//4. LSTM Unit 1.
			vector<Blob<Dtype>*> lstm_bottom_vec(grid_dim_ * 2, NULL);
			vector<Blob<Dtype>*> lstm_top_vec(grid_dim_ * 2, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					lstm_bottom_vec[i * 2] = zero_memory_.get();
				}
				else
				{
					lstm_bottom_vec[i * 2] = C_i_[link_idx_[dp][i]][i].get();
				}
				lstm_bottom_vec[i * 2 + 1] = G_h_[dp][i].get();
				lstm_top_vec[i * 2] = C_i_[dp][i].get();
				lstm_top_vec[i * 2 + 1] = H_i_[dp][i].get();
			}
			lstm_unit_h_->Backward(lstm_top_vec, 
				vector<bool>(grid_dim_ * 2, true), lstm_bottom_vec);
			//3. forward gate.
			for (int i = 0; i < grid_dim_; i++)
			{
				const vector<Blob<Dtype>*> ip_bottom_vec(1, XH_h_k_[dp][i].get());
				const vector<Blob<Dtype>*> ip_top_vec(1, G_h_[dp][i].get());
				ip_xh_h_[i]->Backward(ip_top_vec, 
					vector<bool>(1, true), ip_bottom_vec);
			}
			//2.5. split xh_h.
			const vector<Blob<Dtype>*> split_xh_bottom_vec(1, XH_h_[dp].get());
			vector<Blob<Dtype>*> split_xh_top_vec(grid_dim_, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				split_xh_top_vec[i] = XH_h_k_[dp][i].get();
			}
			split_xh_h_->Backward(split_xh_top_vec, 
				vector<bool>(1, true), split_xh_bottom_vec);
			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom_vec(1 + grid_dim_, NULL);
			concat_bottom_vec[0] = X_1_[dp].get();
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					concat_bottom_vec[1 + i] = zero_memory_.get();
				}
				else
				{
					concat_bottom_vec[1 + i] = H_i_1_[link_idx_[dp][i]][i].get();
				}
			}
			const vector<Blob<Dtype>*> concat_top_vec(1, XH_h_[dp].get());
			concat_h_->Backward(concat_top_vec, 
				vector<bool>(1 + grid_dim_, true), concat_bottom_vec);
		}
		// 1.5 split
		for (int i = 0; i < num_seq_; i++)
		{
			const vector<Blob<Dtype>*> split2_bottom_vec(1, X_[i].get());
			const vector<Blob<Dtype>*> split2_top_vec{ X_1_[i].get(), X_2_[i].get() };
			split_x_->Backward(split2_top_vec, 
				vector<bool>(1, true), split2_bottom_vec);
		}
		// 1. copy bottom to X_.
		if (propagate_down[0])
		{
			int idx = 0;
			Dtype* bottom_data = bottom[0]->mutable_cpu_diff();
			for (int n = 0; n < num; n++)
			{
				for (int c = 0; c < channels; c++)
				{
					for (int d = 0; d < num_seq_; d++)
					{
						const Dtype* X_data = X_[d]->cpu_diff();
						*(bottom_data++) = X_data[idx];
					}
					idx++;
				}
			}
		}
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::CalculateOrder(const vector<Blob<Dtype>*>& bottom)
	{
		link_idx_.resize(num_seq_);
		for (int i = 0; i < num_seq_; i++)
		{
			link_idx_[i].resize(grid_dim_);
			int pi = i;
			for (int j = grid_dim_ - 1; j >= 0; j--)
			{
				if (!reverse_[j])
				{
					if (pi % bottom[0]->shape(j + 2) == 0)
					{
						link_idx_[i][j] = -1;
					}
					else
					{
						link_idx_[i][j] = i - bottom[0]->count(2 + j + 1);
					}
				}
				else
				{
					if (pi % bottom[0]->shape(j + 2) == bottom[0]->shape(j + 2) - 1)
					{
						link_idx_[i][j] = -1;
					}
					else
					{
						link_idx_[i][j] = i + bottom[0]->count(2 + j + 1);
					}
				}
				pi /= bottom[0]->shape(j + 2);
			}
		}
		vector<bool> finish(num_seq_, false);
		order_ = vector<int>(num_seq_, -1);
		int o_t = 0;
		while (o_t != num_seq_)
		{
			for (int i = 0; i < num_seq_; i++)
			{
				if (finish[i])
				{
					continue;
				}
				bool flag = true;
				for (int j = 0; j < grid_dim_; j++)
				{
					if (link_idx_[i][j] >= 0 && !finish[link_idx_[i][j]])
					{
						flag = false;
					}
				}
				if (flag)
				{
					finish[i] = true;
					order_[o_t++] = i;
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(GridLSTMLayer);
#endif

	INSTANTIATE_CLASS(GridLSTMLayer);
	REGISTER_LAYER_CLASS(GridLSTM);
}  // namespace caffe