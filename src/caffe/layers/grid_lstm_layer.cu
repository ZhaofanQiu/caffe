
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
	__global__ void Blob2Xd(int n, const Dtype* blob, int d, 
		int num_seq, Dtype* xd) {
		CUDA_KERNEL_LOOP(index, n) {
			xd[index] = blob[index * num_seq + d];
		}
	}

	template <typename Dtype>
	__global__ void Xd2Blob(int n, const Dtype* xd, int d,
		int num_seq, Dtype* blob) {
		CUDA_KERNEL_LOOP(index, n) {
			blob[index * num_seq + d] = xd[index];
		}
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		// 1. copy bottom to X_.
		const Dtype* bottom_data = bottom[0]->gpu_data();
		for (int d = 0; d < num_seq_; d++)
		{
			Dtype* X_data = X_[d]->mutable_gpu_data();
			Blob2Xd<Dtype> << <CAFFE_GET_BLOCKS(num * channels), CAFFE_CUDA_NUM_THREADS >> >(
				num * channels, bottom_data, d, num_seq_, X_data);
		}
		// 1.5 split
		for (int i = 0; i < num_seq_; i++)
		{
			vector<Blob<Dtype>*> split2_bottom_vec(1, X_[i].get());
			vector<Blob<Dtype>*> split2_top_vec(3, NULL);
			split2_top_vec[0] = X_1_[i].get();
			split2_top_vec[1] = X_2_[i].get();
			split2_top_vec[2] = X_3_[i].get();
			split_x_->Forward(split2_bottom_vec, split2_top_vec);
		}
		// For all sequence run lstm1.
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
			vector<Blob<Dtype>*> concat_top_vec(1, XH_h_[dp].get());
			concat_h_->Forward(concat_bottom_vec, concat_top_vec);
			//2.5. split xh_h.
			vector<Blob<Dtype>*> split_xh_bottom_vec(1, XH_h_[dp].get());
			vector<Blob<Dtype>*> split_xh_top_vec(grid_dim_, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				split_xh_top_vec[i] = XH_h_k_[dp][i].get();
			}
			split_xh_h_->Forward(split_xh_bottom_vec, split_xh_top_vec);
			//3. forward gate.
			for (int i = 0; i < grid_dim_; i++)
			{
				vector<Blob<Dtype>*> ip_bottom_vec(1, XH_h_k_[dp][i].get());
				vector<Blob<Dtype>*> ip_top_vec(1, G_h_[dp][i].get());
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
				vector<Blob<Dtype>*> split1_bottom_vec(1, H_i_[dp][i].get());
				vector<Blob<Dtype>*> split1_top_vec(2, NULL);
				split1_top_vec[0] = H_i_1_[dp][i].get();
				split1_top_vec[1] = H_i_2_[dp][i].get();
				split_h_->Forward(split1_bottom_vec, split1_top_vec);
			}
		}
		// For all sequence run lstm2.
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
			vector<Blob<Dtype>*> concat_top_vec(1, XH_x_[dp].get());
			concat_x_->Forward(concat_bottom_vec, concat_top_vec);
			//6. forward gate.
			vector<Blob<Dtype>*> ip_bottom_vec(1, XH_x_[dp].get());
			vector<Blob<Dtype>*> ip_top_vec(1, G_x_[dp].get());
			ip_xh_x_->Forward(ip_bottom_vec, ip_top_vec);
			//7. LSTM Unit 2.
			vector<Blob<Dtype>*> lstm_bottom_vec(2, NULL);
			vector<Blob<Dtype>*> lstm_top_vec(2, NULL);
			lstm_bottom_vec[0] = X_3_[dp].get();
			lstm_bottom_vec[1] = G_x_[dp].get();
			lstm_top_vec[0] = X_c_[dp].get();
			lstm_top_vec[1] = X_h_[dp].get();
			lstm_unit_x_->Forward(lstm_bottom_vec, lstm_top_vec);
		}
		//8. copy top.
		Dtype* top_data = top[0]->mutable_gpu_data();
		for (int d = 0; d < num_seq_; d++)
		{
			const Dtype* X_data = X_h_[d]->gpu_data();
			Xd2Blob<Dtype> << <CAFFE_GET_BLOCKS(num * channels), CAFFE_CUDA_NUM_THREADS >> >(
				num * channels, X_data, d, num_seq_, top_data);
		}
	}

	template <typename Dtype>
	void GridLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		const int num = bottom[0]->shape(0);
		const int channels = bottom[0]->shape(1);
		
		//8. copy top.
		const Dtype* top_diff = top[0]->gpu_diff();
		for (int d = 0; d < num_seq_; d++)
		{
			Dtype* X_diff = X_h_[d]->mutable_gpu_diff();
			Blob2Xd<Dtype> << <CAFFE_GET_BLOCKS(num * channels), CAFFE_CUDA_NUM_THREADS >> >(
				num * channels, top_diff, d, num_seq_, X_diff);
		}
		// For all sequence run lstm2.
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[num_seq_ - 1 - d];
			//7. LSTM Unit 2.
			vector<Blob<Dtype>*> lstm_bottom_vec(2, NULL);
			vector<Blob<Dtype>*> lstm_top_vec(2, NULL);
			vector<bool> lstm_prop(2, true);
			lstm_bottom_vec[0] = X_3_[dp].get();
			lstm_bottom_vec[1] = G_x_[dp].get();
			lstm_top_vec[0] = X_c_[dp].get();
			lstm_top_vec[1] = X_h_[dp].get();
			lstm_unit_x_->Backward(lstm_top_vec, lstm_prop, lstm_bottom_vec);
			//6. forward gate.
			vector<Blob<Dtype>*> ip_bottom_vec(1, XH_x_[dp].get());
			vector<Blob<Dtype>*> ip_top_vec(1, G_x_[dp].get());
			vector<bool> ip_prop(1, true);
			ip_xh_x_->Backward(ip_top_vec, ip_prop, ip_bottom_vec);
			//5. concat x & h_t
			vector<Blob<Dtype>*> concat_bottom_vec(1 + grid_dim_, NULL);
			vector<bool> concat_prop(1 + grid_dim_, true);
			concat_bottom_vec[0] = X_2_[dp].get();
			concat_prop[0] = true;
			for (int i = 0; i < grid_dim_; i++)
			{
				concat_bottom_vec[1 + i] = H_i_2_[dp][i].get();
				concat_prop[1 + i] = true;
			}
			vector<Blob<Dtype>*> concat_top_vec(1, XH_x_[dp].get());
			concat_x_->Backward(concat_top_vec, concat_prop, concat_bottom_vec);
		}

		// For all sequence run lstm1.
		for (int d = 0; d < num_seq_; d++)
		{
			int dp = order_[num_seq_ - 1 - d];
			// 4.5 split
			for (int i = 0; i < grid_dim_; i++)
			{
				vector<Blob<Dtype>*> split1_bottom_vec(1, H_i_[dp][i].get());
				vector<bool> split1_prop(1, true);
				vector<Blob<Dtype>*> split1_top_vec(2, NULL);
				split1_top_vec[0] = H_i_1_[dp][i].get();
				split1_top_vec[1] = H_i_2_[dp][i].get();
				split_h_->Backward(split1_top_vec, split1_prop, split1_bottom_vec);
			}
			//4. LSTM Unit 1.
			vector<Blob<Dtype>*> lstm_bottom_vec(grid_dim_ * 2, NULL);
			vector<bool> lstm_prop(grid_dim_ * 2, true);
			vector<Blob<Dtype>*> lstm_top_vec(grid_dim_ * 2, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					lstm_bottom_vec[i * 2] = zero_memory_.get();
					lstm_prop[i * 2] = false;
				}
				else
				{
					lstm_bottom_vec[i * 2] = C_i_[link_idx_[dp][i]][i].get();
					lstm_prop[i * 2] = true;
				}
				lstm_bottom_vec[i * 2 + 1] = G_h_[dp][i].get();
				lstm_prop[i * 2 + 1] = true;
				lstm_top_vec[i * 2] = C_i_[dp][i].get();
				lstm_top_vec[i * 2 + 1] = H_i_[dp][i].get();
			}
			lstm_unit_h_->Backward(lstm_top_vec, lstm_prop, lstm_bottom_vec);
			//3. forward gate.
			for (int i = 0; i < grid_dim_; i++)
			{
				vector<Blob<Dtype>*> ip_bottom_vec(1, XH_h_k_[dp][i].get());
				vector<bool> ip_prop(1, true);
				vector<Blob<Dtype>*> ip_top_vec(1, G_h_[dp][i].get());
				ip_xh_h_[i]->Backward(ip_top_vec, ip_prop, ip_bottom_vec);
			}
			//2.5. split xh_h.
			vector<Blob<Dtype>*> split_xh_bottom_vec(1, XH_h_[dp].get());
			vector<bool> split_xh_prop(1, true);
			vector<Blob<Dtype>*> split_xh_top_vec(grid_dim_, NULL);
			for (int i = 0; i < grid_dim_; i++)
			{
				split_xh_top_vec[i] = XH_h_k_[dp][i].get();
			}
			split_xh_h_->Backward(split_xh_top_vec, split_xh_prop, split_xh_bottom_vec);
			//2. concat x & h_t-1.
			vector<Blob<Dtype>*> concat_bottom_vec(1 + grid_dim_, NULL);
			concat_bottom_vec[0] = X_1_[dp].get();
			vector<bool> concat_prop(1 + grid_dim_, true);
			concat_prop[0] = true;
			for (int i = 0; i < grid_dim_; i++)
			{
				if (link_idx_[dp][i] < 0)
				{
					concat_bottom_vec[1 + i] = zero_memory_.get();
					concat_prop[1 + i] = false;
				}
				else
				{
					concat_bottom_vec[1 + i] = H_i_1_[link_idx_[dp][i]][i].get();
					concat_prop[1 + i] = true;
				}
			}
			vector<Blob<Dtype>*> concat_top_vec(1, XH_h_[dp].get());
			concat_h_->Backward(concat_top_vec, concat_prop, concat_bottom_vec);
		}
		// 1.5 split
		for (int i = 0; i < num_seq_; i++)
		{
			vector<Blob<Dtype>*> split2_bottom_vec(1, X_[i].get());
			vector<bool> split2_prop(1, true);
			vector<Blob<Dtype>*> split2_top_vec(3, NULL);
			split2_top_vec[0] = X_1_[i].get();
			split2_top_vec[1] = X_2_[i].get();
			split2_top_vec[2] = X_3_[i].get();
			split_x_->Backward(split2_top_vec, split2_prop, split2_bottom_vec);
		}
		// 1. copy bottom to X_.
		if (propagate_down[0])
		{
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			for (int d = 0; d < num_seq_; d++)
			{
				const Dtype* X_diff = X_[d]->gpu_diff();
				Xd2Blob<Dtype> << <CAFFE_GET_BLOCKS(num * channels), CAFFE_CUDA_NUM_THREADS >> >(
					num * channels, X_diff, d, num_seq_, bottom_diff);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GridLSTMLayer);
}  // namespace caffe