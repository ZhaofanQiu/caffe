
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/9/11
** desc： convert_c3d_model_and_mean tool
*********************************************************************************/

#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using caffe::BlobProto;
using caffe::NetParameter;

void resize_blob_proto(BlobProto* proto, vector<int> shape)
{
	LOG(INFO) << "Convert layer";
	int count = 1;
	for (int i = 0; i < shape.size(); i++)
	{
		count *= shape[i];
	}
	CHECK_EQ(count, proto->data_size());
	proto->clear_shape();
	for (int i = 0; i < shape.size(); i++)
	{
		proto->mutable_shape()->add_dim(shape[i]);
	}
}

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc != 2)
	{
		cout << "usage: convert_c3d_model_and_mean.exe c3d_model" << endl;
		return 0;
	}
	std::string model_path = argv[1];

	//convert model
	NetParameter param;
	caffe::ReadProtoFromBinaryFile(model_path, &param);
	for (int i = 0; i < param.layer_size(); ++i) 
	{
		caffe::LayerParameter* layer = param.mutable_layer(i);
		if (layer->name() == "fc6-1")
		{
			vector<int> fc6_shape0(4, 0);
			fc6_shape0[0] = 4096;
			fc6_shape0[1] = 512;
			fc6_shape0[2] = 4;
			fc6_shape0[3] = 4;
			resize_blob_proto(layer->mutable_blobs(0), fc6_shape0);
		}
		if (layer->name() == "fc7-1")
		{
			vector<int> fc7_shape0(4, 0);
			fc7_shape0[0] = 4096;
			fc7_shape0[1] = 4096;
			fc7_shape0[2] = 1;
			fc7_shape0[3] = 1;
			resize_blob_proto(layer->mutable_blobs(0), fc7_shape0);
		}
		if (layer->name() == "fc8-1")
		{
			vector<int> fc8_shape0(4, 0);
			fc8_shape0[0] = 487;
			fc8_shape0[1] = 4096;
			fc8_shape0[2] = 1;
			fc8_shape0[3] = 1;
			resize_blob_proto(layer->mutable_blobs(0), fc8_shape0);
		}
	}
	caffe::WriteProtoToBinaryFile(param, "fc_conv3d_deepnetA_sport1m_iter_1900000");
	return 0;
}