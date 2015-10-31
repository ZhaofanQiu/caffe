//added by Qing Li, 2014-12-13
// This program converts a set of features to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_featureset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the features, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.fisher 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

#define CHANNELS 40012800
#define LENGTH 1
#define HEIGHT 1
#define WIDTH 1

float buf[CHANNELS * LENGTH * HEIGHT * WIDTH];

bool ReadOnefileFeatureToVolumeDatum(FILE* file, VolumeDatum* datum)
{
	if (file == NULL)
	{
		LOG(ERROR) << "Could not open or find file ";
		return false;
	}
	int num_dims;
	if (!fread(&num_dims, sizeof(__int32), 1, file))
	{
		return false;
	}
	CHECK(num_dims == 4 || num_dims == 5);
	
	int* dims = new int[num_dims];
	fread(dims, sizeof(__int32), num_dims, file);
	if (num_dims == 5)
	{
		CHECK_EQ(dims[1], CHANNELS);
		CHECK_EQ(dims[2], LENGTH);
		CHECK_EQ(dims[3], HEIGHT);
		CHECK_EQ(dims[4], WIDTH);

		datum->set_channels(dims[1]);
		datum->set_length(dims[2]);
		datum->set_height(dims[3]);
		datum->set_width(dims[4]);
		datum->set_label(0);
	}
	else
	{
		CHECK_EQ(dims[1], CHANNELS);
		CHECK_EQ(dims[2], HEIGHT);
		CHECK_EQ(dims[3], WIDTH);

		datum->set_channels(dims[1]);
		datum->set_length(1);
		datum->set_height(dims[2]);
		datum->set_width(dims[3]);
		datum->set_label(0);
	}


	datum->clear_data();
	datum->clear_float_data();
	CHECK_EQ(CHANNELS * LENGTH * HEIGHT * WIDTH, fread(buf, sizeof(float), CHANNELS * LENGTH * HEIGHT * WIDTH, file)) << "file not full read";

	auto datum_ptr = datum->mutable_float_data();
	for (int dim = 0; dim < CHANNELS * LENGTH * HEIGHT * WIDTH; dim++)
	{
		datum_ptr->Add(buf[dim]);
	}
	return true;

	delete dims;
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc != 3) {
		printf("Convert a set of features to the leveldb format used\n"
			"as input for Caffe.\n"
			"Usage:\n"
			"    convert_feature onefile DB_NAME \n");
		return 1;
	}
	FILE* file;
	file = fopen(argv[1], "rb");

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO) << "Opening leveldb " << argv[2];
	leveldb::Status status = leveldb::DB::Open(
		options, argv[2], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[2];

	string root_folder(argv[1]);
	VolumeDatum datum;
	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;
	/*std::stringstream str2int;
	str2int<<argv[4];
	int channel;
	str2int>>channel;*/

	int line_id = 0;
	while (!feof(file))
	{
		if (!ReadOnefileFeatureToVolumeDatum(file, &datum))
		{
			continue;
		}
		line_id++;
		if (!data_size_initialized) {
			data_size = datum.channels() * datum.length() * datum.height() * datum.width();
			data_size_initialized = true;
		}
		else {
			//const string& data = datum.float_data();
			CHECK_EQ(datum.float_data().size(), data_size) << "Incorrect data field size "
				<< datum.float_data().size();
		}
		// sequential
		_snprintf(key_cstr, kMaxKeyLength, "%08d", line_id);
		string value;
		// get the value
		datum.SerializeToString(&value);
		batch->Put(string(key_cstr), value);
		if (++count % 100 == 0) {
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR) << "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	// write the last batch
	if (count % 100 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR) << "Processed " << count << " files.";
	}
	fclose(file);

	delete batch;
	delete db;
	return 0;
}
