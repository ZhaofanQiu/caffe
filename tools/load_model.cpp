
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
	if (argc != 2)
	{
		cout << "usage: load_model.exe model" << endl;
		return 0;
	}

	std::string model_path = argv[1];

	//load model
	NetParameter param;
	caffe::ReadProtoFromBinaryFile(model_path, &param);
	for (int i = 0; i < param.layer_size(); ++i) 
	{
		const caffe::LayerParameter layer = param.layer(i);
		if (layer.name() == "all_score11")
		{
			FILE* file = fopen("output.bin", "wb");
			cout << layer.blobs(0).data_size() << endl;
			for (int i = 0; i < layer.blobs(0).data_size(); ++i)
			{
				const float temp = layer.blobs(0).data(i);
				fwrite(&temp, sizeof(float), 1, file);
			}
			cout << layer.blobs(1).data_size() << endl;
			for (int i = 0; i < layer.blobs(1).data_size(); ++i)
			{
				const float temp = layer.blobs(1).data(i);
				fwrite(&temp, sizeof(float), 1, file);
			}
			fclose(file);
		}
	}
	return 0;
}