
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： OnefileData layer
*********************************************************************************/

#ifndef CAFFE_ONEFILE_DATA_LAYER_HPP_
#define CAFFE_ONEFILE_DATA_LAYER_HPP_

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
#include "caffe/layers/base_data_layer.hpp"

namespace caffe {
	/**
	* @brief Layer to read video data from list.
	*/
	template <typename Dtype>
	class OnefileDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit OnefileDataLayer(const LayerParameter& param)
			: BasePrefetchingDataLayer<Dtype>(param) {}
		virtual ~OnefileDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "OnefileData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void load_batch(Batch<Dtype>* batch);

		string source_;
		FILE* fp_;

		vector<int> top_shape_;
	};
}  // namespace caffe

#endif  // CAFFE_ONEFILE_DATA_LAYER_HPP_
