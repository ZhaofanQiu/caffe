
/********************************************************************************
** Copyright(c) 2015 USTC & MSRA All Rights Reserved.
** auth£º Zhaofan Qiu
** mail£º zhaofanqiu@gmail.com
** date£º 2015/12/13
** desc£º Caffe-video common
*********************************************************************************/

#include <string>
#include <utility>
#include <vector>

#include "caffe/video/video_common.hpp"

namespace caffe {
	using namespace std;
	vector<int> video_shape(int num, int channels, int length, int height, int width)
	{
		vector<int> shape(5, 0);
		shape[0] = num;
		shape[1] = channels;
		shape[2] = length;
		shape[3] = height;
		shape[4] = width;
		return shape;
	}

	void save_data_to_file(const int count, const float* data, const string filename)
	{
		FILE* fp = fopen(filename.c_str(), "w");
		for (int i = 0; i < count; ++i)
		{
			if (i != 0 && i % 100 == 0)
			{
				fprintf(fp, "\n");
			}
			fprintf(fp, "%f\t", data[i]);
		}
		fclose(fp);
	}

	void save_data_to_file(const int count, const double* data, const string filename)
	{
		FILE* fp = fopen(filename.c_str(), "w");
		for (int i = 0; i < count; ++i)
		{
			if (i != 0 && i % 100 == 0)
			{
				fprintf(fp, "\n");
			}
			fprintf(fp, "%lf\t", data[i]);
		}
		fclose(fp);
	}

	void wait_key()
	{
		printf("Press enter...\n");
		char c;
		scanf("%c", &c);
	}
}  // namespace caffe

