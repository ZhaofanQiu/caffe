
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth£º Zhaofan Qiu
** mail£º zhaofanqiu@gmail.com
** date£º 2015/12/13
** desc£º Caffe-video common head
*********************************************************************************/

#ifndef CAFFE_VIDEO_COMMON_HPP_
#define CAFFE_VIDEO_COMMON_HPP_

#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace caffe {

	vector<int> video_shape(int num, int channels = 0, int length = 0, int height = 0, int width = 0);
	void save_data_to_file(const int count, const float* data, const string filename);
	void save_data_to_file(const int count, const double* data, const string filename);
	void wait_key();
}  // namespace caffe

#endif  // CAFFE_VIDEO_COMMON_HPP_

