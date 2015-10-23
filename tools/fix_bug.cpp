
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
#include "boost/shared_ptr.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

using caffe::BlobProto;
using caffe::NetParameter;
using caffe::BlobShape;
using caffe::LayerParameter;

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc != 2)
	{
		cout << "usage: add_c3d_softmax.exe c3d_model" << endl;
		return 0;
	}

	std::string model_path = argv[1];

	//convert model
	NetParameter param;
	caffe::ReadProtoFromBinaryFile(model_path, &param);

	for (int i = 0; i < param.layer_size(); i++)
	{
		LayerParameter* layer = param.mutable_layer(i);
		if (layer->name() == "all_score11")
		{
			cout << "fixed" << endl;
			BlobProto* w1 = layer->mutable_blobs(0);
			w1->clear_shape();
			w1->mutable_shape()->add_dim(11);
			w1->mutable_shape()->add_dim(66);
			w1->mutable_shape()->add_dim(1);
			w1->mutable_shape()->add_dim(1);
		}
	}
	caffe::WriteProtoToBinaryFile(param, model_path);
	return 0;
}