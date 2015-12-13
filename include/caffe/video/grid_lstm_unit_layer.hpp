
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： GridLSTMUnit layer
*********************************************************************************/

#ifndef CAFFE_GRID_LSTM_UNIT_LAYER_HPP_
#define CAFFE_GRID_LSTM_UNIT_LAYER_HPP_

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

namespace caffe {
	/**
	* @brief A helper for LSTMLayer: computes a single timestep of the
	*        non-linearity of the LSTM, producing the updated cell and hidden
	*        states.
	*/
	template <typename Dtype>
	class GridLSTMUnitLayer : public Layer<Dtype> {
	public:
		explicit GridLSTMUnitLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "GridLSTMUnit"; }

		virtual inline int MinNumBottomBlobs() const { return 2; }
		virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	protected:
		/**
		* @param bottom input Blob vector (length, 2 * input_num)
		*   -# @f$ (1 \times N \times D) @f$
		*      the previous timestep cell state @f$ c_t-1 @f$
		*   -# @f$ (1 \times N \times 4D) @f$
		*      the "gate inputs" @f$ [i_t', f_t', o_t', g_t'] @f$
		* @param top output Blob vector (length, input_num * 2)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated cell state @f$ c_t @f$, computed as:
		*          i_t := \sigmoid[i_t']
		*          f_t := \sigmoid[f_t']
		*          o_t := \sigmoid[o_t']
		*          g_t := \tanh[g_t']
		*          c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
		*   -# @f$ (1 \times N \times D) @f$
		*      the updated hidden state @f$ h_t @f$, computed as:
		*          h_t := o_t .* \tanh[c_t]
		*/
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		/// @brief The hidden and output dimension.
		int input_num_;
		int hidden_dim_;
		vector<shared_ptr<Blob<Dtype> > > X_acts_;
	};
}  // namespace caffe

#endif  // CAFFE_GRID_LSTM_UNIT_LAYER_HPP_
