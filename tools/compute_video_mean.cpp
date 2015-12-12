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

#define CHANNELS 3 
#define NEW_LENGTH 17
#define NEW_HEIGHT 240
#define NEW_WIDTH 320 

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

  gflags::SetUsageMessage("Compute the mean_file of a set of video given by"
        " a list\n"
        "Usage:\n"
        "    compute_video_mean list_file num_sample output_file \n");

  if (argc != 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
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

  BlobProto sum_blob;
  // load first datum

  BlobShape* shape = sum_blob.mutable_shape();
  shape->clear_dim();
  shape->add_dim(1);
  shape->add_dim(CHANNELS);
  shape->add_dim(NEW_LENGTH);
  shape->add_dim(NEW_HEIGHT);
  shape->add_dim(NEW_WIDTH);

  const int data_size = 1 * CHANNELS * NEW_LENGTH * NEW_HEIGHT * NEW_WIDTH;
  for (int i = 0; i < data_size; ++i) {
	  sum_blob.add_data(0.);
  }

  int num_sample = atoi(argv[2]);
  VolumeDatum datum;
  for (int ri = 0; ri < num_sample; ri++)
  {
	  srand(time(NULL));
	  const int idx = rand() % count;
	  LOG(INFO) << "Processing " << file_list_[idx];
	  CHECK(ReadVideoToVolumeDatum(file_list_[idx].c_str(), -1, label_list_[idx],
		  NEW_LENGTH, NEW_HEIGHT, NEW_WIDTH, 1, &datum));
	  const string& data = datum.data();
	  for (int i = 0; i < data_size; i++)
	  {
		  sum_blob.set_data(i, sum_blob.data(i) + static_cast<uint8_t>(data[i]));
	  }
	  if (ri % 1000 == 0) {
		  LOG(INFO) << "Processed " << ri + 1 << " files.";
	  }
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / num_sample);
  }
  // Write to disk
  LOG(INFO) << "Write to " << argv[3];
  WriteProtoToBinaryFile(sum_blob, argv[3]);

  for (int i = 0; i < sum_blob.data_size(); ++i) {
	  printf("%f ", sum_blob.data(i));
  }
  return 0;
}
