
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： VolumeData layer
*********************************************************************************/

#ifndef CAFFE_VOLUME_DATA_LAYER_HPP_
#define CAFFE_VOLUME_DATA_LAYER_HPP_

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
	template <typename Dtype>
	class VolumeDataLayer : public BasePrefetchingDataLayer<Dtype> {
	public:
		explicit VolumeDataLayer(const LayerParameter& param);
		virtual ~VolumeDataLayer();
		virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// DataLayer uses DataReader instead for sharing for parallelism
		virtual inline const char* type() const { return "VolumeData"; }
		virtual inline int ExactNumBottomBlobs() const { return 0; }
		virtual inline int MinTopBlobs() const { return 1; }
		virtual inline int MaxTopBlobs() const { return 2; }

	protected:
		shared_ptr<leveldb::DB> db_;
		shared_ptr<leveldb::Iterator> iter_;
		shared_ptr<Caffe::RNG> prefetch_rng_;
		virtual unsigned int PrefetchRand();
		virtual void load_batch(Batch<Dtype>* batch);
		vector<int> top_shape_;
		Blob<Dtype> data_mean_;
		int origin_width_;
		int origin_height_;
	};
}  // namespace caffe

#endif  // CAFFE_VOLUME_DATA_LAYER_HPP_
