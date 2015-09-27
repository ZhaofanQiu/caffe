
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

void convert_to_3d_conv_blob(BlobProto* proto)
{
	LOG(INFO) << "Convert layer";
	const BlobShape shape = proto->shape();
	CHECK(shape.dim_size() == 4 || shape.dim_size() == 5);
	int count = 1;
	for (int i = 0; i < shape.dim_size(); i++)
	{
		count *= shape.dim(i);
	}

	if (shape.dim_size() == 4)
	{
		proto->mutable_shape()->add_dim(3);
		float* buffer = new float[count];
		for (int i = 0; i < count; i++)
		{
			buffer[i] = proto->data(i);
		}
		proto->clear_data();
		for (int i = 0; i < count / (3 * 3); i++)
		{
			for (int j = 0; j < 3; j++)
			{
				for (int k = 0; k < 3 * 3; k++)
				{
					proto->add_data(buffer[i * 3 * 3 + k] / 3);
				}
			}
		}
		delete buffer;
	}
}

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc < 3)
	{
		cout << "usage: convert_vgg_to_3d.exe vgg_model [layers]" << endl;
		return 0;
	}

	std::string model_path = argv[1];

	//convert model
	NetParameter param;
	caffe::ReadProtoFromBinaryFile(model_path, &param);
	for (int i = 0; i < param.layer_size(); ++i) 
	{
		caffe::LayerParameter* layer = param.mutable_layer(i);
		for (int j = 0; j < argc - 2; j++)
		{
			if (layer->name() == argv[j + 2])
			{
				convert_to_3d_conv_blob(layer->mutable_blobs(0));
			}
		}
	}
	std::string out_name = "vgg_3d_model";
	for (int j = 0; j < argc - 2; j++)
	{
		out_name = out_name + "_" + argv[j + 2];
	}
	caffe::WriteProtoToBinaryFile(param, out_name);
	return 0;
}