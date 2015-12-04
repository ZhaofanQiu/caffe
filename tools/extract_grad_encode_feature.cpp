/*
*
*  Copyright (c) 2015, Facebook, Inc. All rights reserved.
*
*  Licensed under the Creative Commons Attribution-NonCommercial 3.0
*  License (the "License"). You may obtain a copy of the License at
*  https://creativecommons.org/licenses/by-nc/3.0/.
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
*  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
*  License for the specific language governing permissions and limitations
*  under the License.
*
*
*/

#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"

#define uint unsigned int

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
	return feature_extraction_pipeline<float>(argc, argv);
}

template<typename Dtype>
bool append_grad_to_binary(FILE* f, Blob<Dtype>* blob)
{
	float *buff;
	if (f == NULL)
		return false;

	buff = blob->mutable_cpu_diff();

	int num_axes_ = blob->num_axes();
	fwrite(&num_axes_, sizeof(int), 1, f);
	for (int i = 0; i < blob->num_axes(); i++)
	{
		int shape_i_ = blob->shape(i);
		fwrite(&shape_i_, sizeof(int), 1, f);
	}
	fwrite(buff, sizeof(Dtype), blob->count(), f);
	return true;
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
	char* net_proto = argv[1];
	char* pretrained_model = argv[2];
	int device_id = atoi(argv[3]);
	uint batch_size = atoi(argv[4]);
	uint num_mini_batches = atoi(argv[5]);
	char* fn_feat = argv[6];

	if (device_id >= 0){
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(device_id);
		LOG(ERROR) << "Using GPU #" << device_id;
	}
	else{
		Caffe::set_mode(Caffe::CPU);
		LOG(ERROR) << "Using CPU";
	}

	boost::shared_ptr<Net<Dtype> > feature_extraction_net(
		new Net<Dtype>(string(net_proto), caffe::TEST));
	feature_extraction_net->CopyTrainedLayersFrom(string(pretrained_model));

	for (int i = 7; i<argc; i++){
		CHECK(feature_extraction_net->param_names_index_[string(argv[i])] >= 0)
			<< "Unknown feature blob name " << string(argv[i])
			<< " in the network " << string(net_proto);
	}

	LOG(ERROR) << "Extracting features for " << num_mini_batches << " batches";

	
	int c = 0;

	vector<Blob<float>*> input_vec;
	int image_index = 0;

	vector<string> outfile = vector<string>(argc - 7, "");
	vector<FILE*> fps = vector<FILE*>(argc - 7, NULL);
	for (int k = 7; k < argc; k++)
	{
		outfile[k - 7] = fn_feat + string(".") + string(argv[k]);
		fps[k - 7] = fopen(outfile[k - 7].c_str(), "wb");
	}
		
	for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
		feature_extraction_net->Forward(input_vec);
		feature_extraction_net->Backward();

		for (int k = 7; k<argc; k++){
			const int owner_net_param_id = 
				feature_extraction_net->param_names_index_[string(argv[k])];
			const pair<int, int>& owner_index =
				feature_extraction_net->param_layer_indices_[owner_net_param_id];
			const int owner_layer_id = owner_index.first;
			const int owner_param_id = owner_index.second;
			Blob<Dtype>* owner_blob =
				feature_extraction_net->layers_[owner_layer_id]->blobs()[owner_param_id].get();

			append_grad_to_binary(fps[k - 7], owner_blob);
		}
		image_index += batch_size;
		if (batch_index % 20 == 0) {
			LOG(ERROR) << "Extracted features of " << image_index <<
				" images.";
		}
		feature_extraction_net->ClearParamDiffs();
	}
	LOG(ERROR) << "Successfully extracted " << image_index << " features!";
	for (int k = 7; k < argc; k++)
	{
		fclose(fps[k - 7]);
	}
	
	return 0;
}
