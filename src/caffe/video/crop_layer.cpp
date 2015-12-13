
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth： Zhaofan Qiu
** mail： zhaofanqiu@gmail.com
** date： 2015/12/13
** desc： Crop layer
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/crop_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->shape(-2), bottom[1]->shape(-2))
			<< "Crop layer first bottom height must be greater";
		CHECK_GE(bottom[0]->shape(-1), bottom[1]->shape(-1))
			<< "Crop layer first bottom width must be greater";
		crop_h_ = round((bottom[0]->shape(-2) - bottom[1]->shape(-2)) / 2);
		crop_w_ = round((bottom[0]->shape(-1) - bottom[1]->shape(-1)) / 2);
	}

	template <typename Dtype>
	void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		vector<int> top_shape = bottom[0]->shape();
		top_shape[top_shape.size() - 2] = bottom[1]->shape(-2);
		top_shape[top_shape.size() - 1] = bottom[1]->shape(-1);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();

		int outer = top[0]->count(0, top[0]->num_axes() - 2);
		int offset_b = bottom[0]->shape(-1) * bottom[0]->shape(-2);
		int offset_t = top[0]->shape(-1) * top[0]->shape(-2);
		caffe_set(top[0]->count(), static_cast<Dtype>(0), top_data);
		for (int o = 0; o < outer; o++)
		{
			for (int h = 0; h < top[0]->shape(-2); ++h) {
				caffe_copy(top[0]->shape(-1),
					bottom_data + (crop_h_ + h) * bottom[0]->shape(-1) + crop_w_,
					top_data + h * top[0]->shape(-1));
			}
			bottom_data += offset_b;
			top_data += offset_t;
		}
	}

	template <typename Dtype>
	void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		int outer = top[0]->count(0, top[0]->num_axes() - 2);
		int offset_b = bottom[0]->shape(-1) * bottom[0]->shape(-2);
		int offset_t = top[0]->shape(-1) * top[0]->shape(-2);
		if (propagate_down[0]) {
			caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
			for (int o = 0; o < outer; o++)
			{
				for (int h = 0; h < top[0]->shape(-2); ++h) {
					caffe_copy(top[0]->shape(-1),
						top_diff + h * top[0]->shape(-1),
						bottom_diff + (crop_h_ + h) * bottom[0]->shape(-1) + crop_w_);
				}
				bottom_diff += offset_b;
				top_diff += offset_t;
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(CropLayer);
#endif

	INSTANTIATE_CLASS(CropLayer);
	REGISTER_LAYER_CLASS(Crop);
}  // namespace caffe
