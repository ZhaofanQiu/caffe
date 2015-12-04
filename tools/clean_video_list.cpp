#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  gflags::SetUsageMessage("Compute the mean_file of a set of video given by"
        " a list\n"
        "Usage:\n"
        "    clean_video_list list_file output_file \n");

  if (argc != 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/clean_video_list");
    return 1;
  }

  std::ifstream infile(argv[1]);
  string filename;
  int label;
  vector<string> file_list_;
  vector<int> label_list_;
  int count = 0;
  while (infile >> filename >> label) {
	  file_list_.push_back(filename);
	  label_list_.push_back(label);
	  count++;
  }
  infile.close();

  VolumeDatum datum;
  std::ofstream outfile(argv[2]);
  for (int ri = 0; ri < file_list_.size(); ri++)
  {
	  cv::VideoCapture cap;
	  cap.open(file_list_[ri]);
	  if (cap.isOpened() && cap.get(CV_CAP_PROP_FRAME_COUNT) >= 30){
		  outfile << file_list_[ri] << " " << label_list_[ri] << std::endl;
	  }
	  else
	  {
		  printf("Wrong video: %s\n", file_list_[ri].c_str());
	  }
  }
  outfile.close();
  return 0;
}
