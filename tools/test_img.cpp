#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/image_io.hpp"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		printf("Usage: FaceRelighting.exe img.jpg\n");
		return 1;
	}
	cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if (!img.data)
	{
		printf("Error img\n");
	}
	else
	{
		cv::imshow(argv[1], img);
		cv::waitKey();
		printf("ok\n");
	}
	return 0;
}
