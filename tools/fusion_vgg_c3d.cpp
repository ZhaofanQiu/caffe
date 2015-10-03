
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

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc != 3)
	{
		cout << "usage: convert_c3d_model_and_mean.exe c3d_model vgg_model" << endl;
		return 0;
	}
	std::string c3d_path = argv[1];
	std::string vgg_path = argv[2];

	//convert model
	NetParameter c3d_param;
	NetParameter vgg_param;
	caffe::ReadProtoFromBinaryFile(c3d_path, &c3d_param);
	caffe::ReadProtoFromBinaryFile(vgg_path, &vgg_param);
	for (int i = 0; i < c3d_param.layer_size(); ++i) 
	{
		caffe::LayerParameter* layer = c3d_param.mutable_layer(i);
		if (layer->name() == "conv1a" ||
			layer->name() == "conv2a" ||
			layer->name() == "conv3a" ||
			layer->name() == "conv3b" ||
			layer->name() == "conv4a" ||
			layer->name() == "conv4b" ||
			layer->name() == "conv5a" ||
			layer->name() == "conv5b")
		{
			cout << layer->name() << endl;
			caffe::LayerParameter* new_layer = vgg_param.add_layer();
			new_layer->CopyFrom(*layer);
		}
	}
	caffe::WriteProtoToBinaryFile(vgg_param, "vgg_c3d_conv_layer");
	return 0;
}