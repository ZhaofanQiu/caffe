
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

using caffe::BlobProto;
using caffe::Blob3DProto;
using caffe::NetParameter3D;
using caffe::NetParameter;

void update_blob_3d_proto(const Blob3DProto& old_3d_proto, BlobProto* new_proto, bool begin = false)
{
	new_proto->clear_shape();
	if (begin || old_3d_proto.num() != 1)
	{
		new_proto->mutable_shape()->add_dim(old_3d_proto.num());
		begin = true;
	}
	if (begin || old_3d_proto.channels() != 1)
	{
		new_proto->mutable_shape()->add_dim(old_3d_proto.channels());
		begin = true;
	}
	if (begin || old_3d_proto.length() != 1)
	{
		new_proto->mutable_shape()->add_dim(old_3d_proto.length());
		begin = true;
	}
	if (begin || old_3d_proto.height() != 1)
	{
		new_proto->mutable_shape()->add_dim(old_3d_proto.height());
		begin = true;
	}
	new_proto->mutable_shape()->add_dim(old_3d_proto.width());

	new_proto->clear_data();
	new_proto->clear_diff();
	for (int i = 0; i < old_3d_proto.data_size(); i++)
	{
		new_proto->add_data(old_3d_proto.data(i));
	}
}

int main(int argc, char** argv) {
	FLAGS_alsologtostderr = 1;
	if (argc != 3)
	{
		cout << "usage: convert_c3d_model_and_mean.exe c3d_model c3d_mean" << endl;
		return 0;
	}
	std::string model_path = argv[1];
	std::string mean_path = argv[2];

	//convert mean
	Blob3DProto old_mean_proto;
	caffe::ReadProtoFromBinaryFileOrDie(mean_path.c_str(), &old_mean_proto);
	BlobProto new_mean_proto;
	update_blob_3d_proto(old_mean_proto, &new_mean_proto, true);
	caffe::WriteProtoToBinaryFile(new_mean_proto, "new_sport1m_train16_128_mean.binaryproto");

	//convert model
	NetParameter3D old_param;
	NetParameter new_param;
	caffe::ReadProtoFromBinaryFile(model_path, &old_param);
	caffe::ReadProtoFromBinaryFile(model_path, &new_param);
	new_param.clear_layers();
	new_param.clear_layer();
	for (int i = 0; i < old_param.layers_size(); ++i) 
	{
		caffe::LayerParameter* new_layer = new_param.add_layer();
		const caffe::LayerParameter3D& old_layer = old_param.layers(i);
		new_layer->Clear();
		for (int i = 0; i < old_layer.bottom_size(); ++i) {
			new_layer->add_bottom(old_layer.bottom(i));
		}
		for (int i = 0; i < old_layer.top_size(); ++i) {
			new_layer->add_top(old_layer.top(i));
		}
		if (old_layer.has_name()) {
			new_layer->set_name(old_layer.name());
		}
		for (int i = 0; i < old_layer.include_size(); ++i) {
			new_layer->add_include()->CopyFrom(old_layer.include(i));
		}
		for (int i = 0; i < old_layer.exclude_size(); ++i) {
			new_layer->add_exclude()->CopyFrom(old_layer.exclude(i));
		}
		for (int i = 0; i < old_layer.blobs_size(); ++i) {
			update_blob_3d_proto(old_layer.blobs(i), new_layer->add_blobs());
		}
		for (int i = 0; i < old_layer.param_size(); ++i) {
			while (new_layer->param_size() <= i) { new_layer->add_param(); }
			new_layer->mutable_param(i)->set_name(old_layer.param(i));
		}
		for (int i = 0; i < old_layer.blobs_lr_size(); ++i) {
			while (new_layer->param_size() <= i) { new_layer->add_param(); }
			new_layer->mutable_param(i)->set_lr_mult(old_layer.blobs_lr(i));
		}
		for (int i = 0; i < old_layer.weight_decay_size(); ++i) {
			while (new_layer->param_size() <= i) { new_layer->add_param(); }
			new_layer->mutable_param(i)->set_decay_mult(
				old_layer.weight_decay(i));
		}
		for (int i = 0; i < old_layer.loss_weight_size(); ++i) {
			new_layer->add_loss_weight(old_layer.loss_weight(i));
		}
		if (old_layer.has_accuracy_param()) {
			new_layer->mutable_accuracy_param()->CopyFrom(
				old_layer.accuracy_param());
		}
		if (old_layer.has_argmax_param()) {
			new_layer->mutable_argmax_param()->CopyFrom(
				old_layer.argmax_param());
		}
		if (old_layer.has_concat_param()) {
			new_layer->mutable_concat_param()->CopyFrom(
				old_layer.concat_param());
		}
		if (old_layer.has_contrastive_loss_param()) {
			new_layer->mutable_contrastive_loss_param()->CopyFrom(
				old_layer.contrastive_loss_param());
		}
		if (old_layer.has_convolution_param()) {
			new_layer->mutable_convolution_param()->CopyFrom(
				old_layer.convolution_param());
		}
		if (old_layer.has_data_param()) {
			new_layer->mutable_data_param()->CopyFrom(
				old_layer.data_param());
		}
		if (old_layer.has_dropout_param()) {
			new_layer->mutable_dropout_param()->CopyFrom(
				old_layer.dropout_param());
		}
		if (old_layer.has_dummy_data_param()) {
			new_layer->mutable_dummy_data_param()->CopyFrom(
				old_layer.dummy_data_param());
		}
		if (old_layer.has_eltwise_param()) {
			new_layer->mutable_eltwise_param()->CopyFrom(
				old_layer.eltwise_param());
		}
		if (old_layer.has_exp_param()) {
			new_layer->mutable_exp_param()->CopyFrom(
				old_layer.exp_param());
		}
		if (old_layer.has_hdf5_data_param()) {
			new_layer->mutable_hdf5_data_param()->CopyFrom(
				old_layer.hdf5_data_param());
		}
		if (old_layer.has_hdf5_output_param()) {
			new_layer->mutable_hdf5_output_param()->CopyFrom(
				old_layer.hdf5_output_param());
		}
		if (old_layer.has_hinge_loss_param()) {
			new_layer->mutable_hinge_loss_param()->CopyFrom(
				old_layer.hinge_loss_param());
		}
		if (old_layer.has_image_data_param()) {
			new_layer->mutable_image_data_param()->CopyFrom(
				old_layer.image_data_param());
		}
		if (old_layer.has_infogain_loss_param()) {
			new_layer->mutable_infogain_loss_param()->CopyFrom(
				old_layer.infogain_loss_param());
		}
		if (old_layer.has_inner_product_param()) {
			new_layer->mutable_inner_product_param()->CopyFrom(
				old_layer.inner_product_param());
		}
		if (old_layer.has_lrn_param()) {
			new_layer->mutable_lrn_param()->CopyFrom(
				old_layer.lrn_param());
		}
		if (old_layer.has_memory_data_param()) {
			new_layer->mutable_memory_data_param()->CopyFrom(
				old_layer.memory_data_param());
		}
		if (old_layer.has_mvn_param()) {
			new_layer->mutable_mvn_param()->CopyFrom(
				old_layer.mvn_param());
		}
		if (old_layer.has_pooling_param()) {
			new_layer->mutable_pooling_param()->CopyFrom(
				old_layer.pooling_param());
		}
		if (old_layer.has_power_param()) {
			new_layer->mutable_power_param()->CopyFrom(
				old_layer.power_param());
		}
		if (old_layer.has_relu_param()) {
			new_layer->mutable_relu_param()->CopyFrom(
				old_layer.relu_param());
		}
		if (old_layer.has_sigmoid_param()) {
			new_layer->mutable_sigmoid_param()->CopyFrom(
				old_layer.sigmoid_param());
		}
		if (old_layer.has_softmax_param()) {
			new_layer->mutable_softmax_param()->CopyFrom(
				old_layer.softmax_param());
		}
		if (old_layer.has_slice_param()) {
			new_layer->mutable_slice_param()->CopyFrom(
				old_layer.slice_param());
		}
		if (old_layer.has_tanh_param()) {
			new_layer->mutable_tanh_param()->CopyFrom(
				old_layer.tanh_param());
		}
		if (old_layer.has_threshold_param()) {
			new_layer->mutable_threshold_param()->CopyFrom(
				old_layer.threshold_param());
		}
		if (old_layer.has_window_data_param()) {
			new_layer->mutable_window_data_param()->CopyFrom(
				old_layer.window_data_param());
		}
		if (old_layer.has_transform_param()) {
			new_layer->mutable_transform_param()->CopyFrom(
				old_layer.transform_param());
		}
		if (old_layer.has_loss_param()) {
			new_layer->mutable_loss_param()->CopyFrom(
				old_layer.loss_param());
		}
	}
	caffe::WriteProtoToBinaryFile(new_param, "new_conv3d_deepnetA_sport1m_iter_1900000");
	return 0;
}