#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_layers.hpp"

namespace caffe {

template <typename Dtype>
void RandomLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();

  CHECK(this->layer_param_.random_loss_param().has_num_label());
  num_label_ = this->layer_param_.random_loss_param().num_label();

  if (this->layer_param_.random_loss_param().has_dropout_ratio())
  {
	  dropout_ratio_ = this->layer_param_.random_loss_param().dropout_ratio();
  }
  else
  {
	  dropout_ratio_ = 0;
  }
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * dropout_ratio_);
  random_idx_.Reshape(vector<int>(1, num_label_));
}

template <typename Dtype>
void RandomLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void RandomLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  unsigned int* mask = random_idx_.mutable_cpu_data();
  caffe_rng_bernoulli(num_label_, (Dtype)dropout_ratio_, mask);
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
	  if ((this->phase_ == TRAIN && mask[label_value]) || (has_ignore_label_ && label_value == ignore_label_)) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  if (this->phase_ == TRAIN)
  {
	  if (normalize_) {
		  top[0]->mutable_cpu_data()[0] = loss / count / (1 - dropout_ratio_);
	  }
	  else {
		  top[0]->mutable_cpu_data()[0] = loss / outer_num_ / (1 - dropout_ratio_);
	  }
  }
  else
  {
	  if (normalize_) {
		  top[0]->mutable_cpu_data()[0] = loss / count;
	  }
	  else {
		  top[0]->mutable_cpu_data()[0] = loss / outer_num_;
	  }
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void RandomLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
	const unsigned int* mask = random_idx_.cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
		if (mask[label_value] || (has_ignore_label_ && label_value == ignore_label_))
		{
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
		caffe_scal(prob_.count(), loss_weight * (1 - dropout_ratio_) / count, bottom_diff);
    } else {
		caffe_scal(prob_.count(), loss_weight * (1 - dropout_ratio_) / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RandomLossLayer);
#endif

INSTANTIATE_CLASS(RandomLossLayer);
REGISTER_LAYER_CLASS(RandomLoss);

}  // namespace caffe
