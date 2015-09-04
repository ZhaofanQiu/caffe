
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/3
** desc： Videodata layer
*********************************************************************************/

#ifndef CAFFE_VIDEO_LAYERS_HPP_
#define CAFFE_VIDEO_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
	/**
	* @brief Layer to read video data from list.
	*/
	template <typename Dtype>
	class VideoDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit VideoDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~VideoDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "VideoData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 2; }

	protected:
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual void load_batch(Batch<Dtype>* batch);
		virtual void ShuffleVideo();
		virtual unsigned int PrefetchRand();

		vector<string> file_list_;
		vector<int> start_frm_list_;
		vector<int> label_list_;
		vector<int> shuffle_index_;
		int lines_id_;
		vector<int> top_shape_;

		Blob<Dtype> data_mean_;
		int origin_width_;
		int origin_height_;
	};
}  // namespace caffe

#endif  // CAFFE_VIDEO_LAYERS_HPP_
