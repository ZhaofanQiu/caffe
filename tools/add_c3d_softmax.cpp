﻿
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
	std::ifstream fin("output.bin", std::ios::ios_base::binary | std::ios::ios_base::in);
	float buffer1[11 * 132];
	fin.read((char*)buffer1, 11 * 132 * sizeof(float));
	float buffer2[11];
	fin.read((char*)buffer2, 11 * sizeof(float));
	fin.close();

	LayerParameter* new_layer = param.add_layer();
	new_layer->set_name("all_score11");
	BlobProto* w1 = new_layer->add_blobs();
	w1->mutable_shape()->add_dim(11);
	w1->mutable_shape()->add_dim(66);
	w1->mutable_shape()->add_dim(1);
	w1->mutable_shape()->add_dim(1);
	for (int i = 0; i < 11; ++i)
	{
		for (int j = 0; j < 66; ++j)
		{
			w1->add_data(buffer1[i * 132 + 66 + j]);
		}
	}
	BlobProto* w2 = new_layer->add_blobs();
	w2->mutable_shape()->add_dim(11);
	for (int i = 0; i < 11; ++i)
	{
		w2->add_data(buffer2[i] / 2);
	}
	caffe::WriteProtoToBinaryFile(param, "c3d_large_softmax_fusion.caffemodel");
	return 0;
}