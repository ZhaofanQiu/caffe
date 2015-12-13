
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： GridLSTM layer
*********************************************************************************/

#ifndef CAFFE_GRID_LSTM_LAYER_HPP_
#define CAFFE_GRID_LSTM_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "boost/scoped_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

#include "caffe/video/video_common.hpp"
#include "caffe/video/grid_lstm_unit_layer.hpp"

#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {
	/**
	* @brief Implementation of Grid LSTM
	*/
	template <typename Dtype>
	class GridLSTMLayer : public Layer<Dtype> {
	public:
		explicit GridLSTMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GridLSTM"; }

		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		void CalculateOrder(const vector<Blob<Dtype>*>& bottom);
		/// @brief The hidden and output dimension.
		vector<bool> reverse_;
		int output_dim_;
		int num_seq_;
		int hidden_dim_;
		int grid_dim_;
		bool bias_term_;
		vector<vector<int> > link_idx_;
		vector<int> order_;
		//Data blobs
		shared_ptr<Blob<Dtype> > zero_memory_;

		//Layers
		// split_x_ layer
		shared_ptr<SplitLayer<Dtype> > split_x_;
		vector<shared_ptr<Blob<Dtype> > > X_;
		vector<shared_ptr<Blob<Dtype> > > X_1_;
		vector<shared_ptr<Blob<Dtype> > > X_2_;

		// split_h_ layer
		shared_ptr<SplitLayer<Dtype> > split_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_1_;
		vector<vector<shared_ptr<Blob<Dtype> > > > H_i_2_;
		
		// split_xh_h_ layer
		shared_ptr<SplitLayer<Dtype> > split_xh_h_;
		vector<shared_ptr<Blob<Dtype> > > XH_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > XH_h_k_;

		// concat_h_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_h_;

		// concat_x_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_x_;
		vector<shared_ptr<Blob<Dtype> > > XH_x_;

		// ip_xh_h_ layer
		vector<shared_ptr<InnerProductLayer<Dtype> > > ip_xh_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > G_h_;

		// lstm_unit_h_ layer
		shared_ptr<GridLSTMUnitLayer<Dtype> > lstm_unit_h_;
		vector<vector<shared_ptr<Blob<Dtype> > > > C_i_;
	};
}  // namespace caffe

#endif  // CAFFE_GRID_LSTM_LAYER_HPP_
