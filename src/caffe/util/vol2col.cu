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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/vol2col.hpp"

namespace caffe {


template <typename Dtype>
__global__ void vol2col_gpu_kernel(const int n, const Dtype* data_im,
    const int length, const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, 
	const int filter_stride, const int filter_stride_l,
	const int length_col, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_out = (index / width_col ) % height_col;
    int l_out = (index / width_col / height_col) % length_col;
    int channel_in = index / width_col / height_col / length_col;
    int channel_out = channel_in * kdepth * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    int l_in = l_out * temporal_stride - temporal_pad;
    
    data_col += ((channel_out * length_col + l_out) * height_col + h_out) * width_col + w_out;
    data_im += ((channel_in * length + l_in) * height + h_in) * width + w_in;
    for (int k = 0; k < kdepth; ++k) {
      for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
          int l = l_in + k * filter_stride_l;
          int h = h_in + i * filter_stride;
          int w = w_in + j * filter_stride;
          *data_col = (l >= 0 && h >= 0 && w >= 0 && h < height && w < width && l < length) ?
              data_im[(k * filter_stride_l * height + i * filter_stride) * width + j * filter_stride] : 0;
          data_col += length_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename Dtype>
void vol2col_gpu(const Dtype* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
	const int temporal_pad, const int stride, const int temporal_stride, 
	const int filter_stride, const int filter_stride_l, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
	int kernel_eff = ksize + (ksize - 1) * (filter_stride - 1);
	int kernel_eff_l = kdepth + (kdepth - 1) * (filter_stride_l - 1);
	int length_col = (length + 2 * temporal_pad - kernel_eff_l) / temporal_stride + 1;
	int height_col = (height + 2 * pad - kernel_eff) / stride + 1;
	int width_col = (width + 2 * pad - kernel_eff) / stride + 1;
  int num_kernels = channels * length_col * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  vol2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, length, height, width, ksize, kdepth, pad, temporal_pad, stride, temporal_stride, 
	  filter_stride, filter_stride_l,
      length_col, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void vol2col_gpu<float>(const float* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, 
	const int filter_stride, const int filter_stride_l_, float* data_col);
template void vol2col_gpu<double>(const double* data_im, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride,
	const int filter_stride, const int filter_stride_l_, double* data_col);

template <typename Dtype>
__global__ void col2vol_gpu_kernel(const int n, const Dtype* data_col,
    const int length, const int height, const int width, const int channels, const int ksize, const int kdepth,
    const int pad, const int temporal_pad, const int stride, const int temporal_stride, 
	const int filter_stride, const int filter_stride_l,
	const int length_col, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int l = (index / width / height) % length + temporal_pad;
    int c = index / (width * height * length);
    // compute the start and end of the output
	int kernel_eff = ksize + (ksize - 1) * (filter_stride - 1);
	int kernel_eff_l = kdepth + (kdepth - 1) * (filter_stride_l - 1);

	int w_col_start = (w < kernel_eff) ? 0 : (w - kernel_eff) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
	int h_col_start = (h < kernel_eff) ? 0 : (h - kernel_eff) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
	int l_col_start = (l < kernel_eff_l) ? 0 : (l - kernel_eff_l) / temporal_stride + 1;
    int l_col_end = min(l / temporal_stride + 1, length_col);
        
	for (int l_col = l_col_start; l_col < l_col_end; ++l_col) {
		if ((l - l_col * temporal_stride) % filter_stride_l == 0)
		{
			for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
				if ((h - h_col * stride) % filter_stride == 0)
				{
					for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
						if ((w - w_col * stride) % filter_stride == 0)
						{
							int c_col = c * kdepth * ksize * ksize
								+ (l - l_col * temporal_stride) / filter_stride_l * ksize * ksize
								+ (h - h_col * stride) / filter_stride * ksize
								+ (w - w_col * stride) / filter_stride;
							val += data_col[((c_col * length_col + l_col) * height_col + h_col) * width_col + w_col];
						}
					}
				}
			}
		}
	}
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride,
	const int filter_stride, const int filter_stride_l, Dtype* data_im) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * length * height * width * channels));
	int kernel_eff = ksize + (ksize - 1) * (filter_stride - 1);
	int kernel_eff_l = kdepth + (kdepth - 1) * (filter_stride_l - 1);
	int length_col = (length + 2 * temporal_pad - kernel_eff_l) / temporal_stride + 1;
	int height_col = (height + 2 * pad - kernel_eff) / stride + 1;
	int width_col = (width + 2 * pad - kernel_eff) / stride + 1;
  int num_kernels = channels * length * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2vol_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, length, height, width, channels, ksize, kdepth, pad, temporal_pad, stride, temporal_stride,
	  filter_stride, filter_stride_l,
      length_col, height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2vol_gpu<float>(const float* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride, 
	const int filter_stride, const int filter_stride_l_, float* data_im);
template void col2vol_gpu<double>(const double* data_col, const int channels, const int length,
    const int height, const int width, const int ksize, const int kdepth, const int pad,
    const int temporal_pad, const int stride, const int temporal_stride,
	const int filter_stride, const int filter_stride_l_, double* data_im);

}  // namespace caffe
