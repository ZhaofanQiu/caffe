
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： MapLSTM layer
*********************************************************************************/

#ifndef CAFFE_MAP_LSTM_LAYER_HPP_
#define CAFFE_MAP_LSTM_LAYER_HPP_

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
#include "caffe/video/map_lstm_unit_layer.hpp"

#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {
	/**
	* @brief Implementation of Map LSTM
	*/
	template <typename Dtype>
	class MapLSTMLayer : public Layer<Dtype> {
	public:
		explicit MapLSTMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}

		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "MapLSTM"; }

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

		/// @brief The hidden and output dimension.
		int hidden_dim_;
		int T_;
		bool bias_term_;
		//Data blobs
		shared_ptr<Blob<Dtype> > zero_memory_;

		//Layers
		// split_h_ layer
		shared_ptr<SplitLayer<Dtype> > split_h_;
		vector<shared_ptr<Blob<Dtype> > > H_;
		vector<shared_ptr<Blob<Dtype> > > H_1_;
		vector<shared_ptr<Blob<Dtype> > > H_2_;

		// concat_h_ layer
		shared_ptr<ConcatLayer<Dtype> > concat_;
		vector<shared_ptr<Blob<Dtype> > > X_;
		vector<shared_ptr<Blob<Dtype> > > XH_;

		// conv_ layer
		shared_ptr<ConvolutionLayer<Dtype> > conv_;
		vector<shared_ptr<Blob<Dtype> > > G_;

		// lstm_unit_h_ layer
		shared_ptr<MapLSTMUnitLayer<Dtype> > lstm_unit_;
		vector<shared_ptr<Blob<Dtype> > > C_;

	};
}  // namespace caffe

#endif  // CAFFE_MAP_LSTM_LAYER_HPP_
