
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

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc != 3)
	{
		cout << "usage: load_model.exe model1 model2" << endl;
		return 0;
	}

	std::string model1_path = argv[1];
	std::string model2_path = argv[2];

	//load model
	NetParameter param1, param2;
	caffe::ReadProtoFromBinaryFile(model1_path, &param1);
	caffe::ReadProtoFromBinaryFile(model2_path, &param2);
	for (int i = 0; i < param1.layer_size(); ++i) 
	{
		const caffe::LayerParameter layer1 = param1.layer(i);
		const caffe::LayerParameter layer2 = param2.layer(i);
		if (layer1.name() == "pool5_lstm")
		{
			for (int i = 0; i < 100; ++i)
			{
				cout << layer1.blobs(0).data(i) << " " << layer2.blobs(0).data(i) << endl;
			}
		}
	}
	return 0;
}