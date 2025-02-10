#include "depthmap_stitcher.hpp"
#include "depthmap_stitcher_enum.hpp"
#include "depthmap_stitcher_group.hpp"
#include "data_imitator.hpp"

#include <glog/logging.h>

#include <gflags/gflags.h>

#include <string>
#include <iostream>


// define CLI arguments and validators
DEFINE_string(type, "stitch", "[stitch|debug]. The program model.");
static bool validate_type(const char* flagname, const std::string& type_str) {
	if (type_str.compare("stitch") != 0 && type_str.compare("debug") != 0)
		return false;
	else
		return true;
}
DEFINE_validator(type, &validate_type);

// The generated data type
DEFINE_string(data_type, "simple", "[simple|rand]. The synthetic data's coefficient is uniformed or random.");
static bool validate_data_type(const char* flagname, const std::string& data_type) {
	if (data_type.compare("simple") != 0 && data_type.compare("rand") != 0)
		return false;
	else
		return true;
}
DEFINE_validator(data_type, &validate_data_type);

DEFINE_string(root_dir, "", "The input and output folder path. ");
static bool validate_root_dir(const char* flagname, const std::string& root_dir) {
	if (root_dir.length() <= 0)
		return false;
	else
		return true;
}
DEFINE_validator(root_dir, &validate_root_dir);

DEFINE_string(method, "enum", "[group|enum]. The stitch type joint optimization or enumerate all depth map pairs.");
static bool validate_method(const char* flagname, const std::string& method_str) {
	if (method_str.compare("group") != 0 && method_str.compare("enum") != 0)
		return false;
	else
		return true;
}
DEFINE_validator(method, &validate_method);

DEFINE_string(filename_list, "", "The input images file name list. --filename_list=\"img0_000.pfm,img0_001.pfm\"");

DEFINE_int32(deform_grid_width, 5, "The depth map alignment deform grid number along the x axis, column number.");
DEFINE_int32(deform_grid_height, 5, "The depth map alignment deform grid number along the y axis, row number.");

// create fake data for debug
// The image (000) is reference image, reference image (000) = target image (001) * scale + offset
void create_debug_data(const std::string& root_dir)
{
	DataImitator di;
	di.json_filepath = root_dir + "align_coeff_gt.json";

	di.depthmap_ref_filepath = root_dir + "img0_depth_000.pfm";
	di.depthmap_tar_filepath = root_dir + "img0_depth_001.pfm";

	di.corr_ref2tar_filepath = root_dir + "img0_corr_000_001.json";
	di.corr_tar2ref_filepath = root_dir + "img0_corr_001_000.json";

	di.depthmap_overlap_ratio = 1.0;
	di.depth_scale = 0.2;
	di.depth_offset = 6.0;
	di.coeff_grid_height = FLAGS_deform_grid_height;
	di.coeff_grid_width = FLAGS_deform_grid_width;

	di.depthmap_width = 50;
	di.depthmap_hight = 80;

	LOG(INFO) << "The reference image (000) is reference image. And target image is 001.\nReference image (001) = target image (000) * scale + offset";

	if (FLAGS_data_type.compare("simple") == 0)
	{
		// 1) generate simple test data
		di.make_aligncoeffs_simple();
		di.make_depthmap_pair_simple();
	}
	else if (FLAGS_data_type.compare("rand") == 0)
	{
		// 2) generate random scale and offset coefficients
		di.make_aligncoeffs_random();
		di.make_depthmap_pair_random();
	}
	di.make_corresponding_json();
	di.output_date();

	LOG(INFO) << di;
}

void create_debug_data_multi(const std::string& root_dir)
{
	DataImitatorMultiImage di;
	di.output_root_dir = "D:/workspace_windows/InstaOmniDepth/InstaOmniDepth_github/code/cpp/bin/Release/";

	// report the data information
	di.report_data_parameters();

	// create data
	di.make_aligncoeffs_random();
	di.make_depthmap_pair();
	di.make_pixel_corresponding();
	di.output_date();
}

// stitch depth maps
void depthmap_stitch(const std::string& root_dir, const std::vector<std::string>& depthmap_filename_list, const std::string& method)
{
	std::shared_ptr<DepthmapStitcher> depthmap_stitcher;
	if (method.compare("enum") == 0)
	{
		depthmap_stitcher = std::make_shared<DepthmapStitcherEnum>(0.1); // set overlap ratio
		depthmap_stitcher->weight_reprojection = 1.0; // re-projection term lambda
		depthmap_stitcher->weight_smooth = 1e-2;     // smooth term lambda
		depthmap_stitcher->weight_scale = 1e-10;      // scale term lambda
	}
	else if (method.compare("group") == 0)
	{
		depthmap_stitcher = std::make_shared<DepthmapStitcherGroup>();
		depthmap_stitcher->weight_reprojection = 1.0; // re-projection term lambda
		depthmap_stitcher->weight_smooth = 1.0;     // smooth term lambda
		depthmap_stitcher->weight_scale = 1e-6;      // scale term lambda
		depthmap_stitcher->depthmap_ref_extidx = 0;  // the reference depth map is the first depth map;
	}
	else
		LOG(ERROR) << "The specified method name " << method << " is wrong.";

	depthmap_stitcher->ceres_max_num_iterations = 300;
	depthmap_stitcher->load_data(root_dir, depthmap_filename_list);
	depthmap_stitcher->initial(FLAGS_deform_grid_width, FLAGS_deform_grid_height);
	depthmap_stitcher->compute_align_coeff();
	depthmap_stitcher->align_depthmap_all();
	depthmap_stitcher->save_aligned_depthmap();
	depthmap_stitcher->save_align_coeff();
	depthmap_stitcher->report_error();
}


int main(int argc, char** argv)
{
	// 0) parser the CLI parameter 
	if (argc <= 2)
		LOG(ERROR) << "The input arguments is less than 3.";
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	const std::string root_dir = FLAGS_root_dir;

	// 1) set-up the runtime environment
	const std::string logfile_path = root_dir + "/depth_map_stitch.log";
	google::SetLogDestination(google::GLOG_INFO, logfile_path.c_str());
	FLAGS_stderrthreshold = google::GLOG_INFO;
	google::InitGoogleLogging(argv[0]);

	LOG(INFO) << "The data root folder :" << root_dir;

	// 2) make mock data or stitch depth maps
	if (FLAGS_type.compare("stitch") == 0)
	{
		std::vector<std::string> depthmap_filename_list;
		size_t start;
		size_t end = 0;
		char delim = ',';
		while ((start = FLAGS_filename_list.find_first_not_of(delim, end)) != std::string::npos)
		{
			end = FLAGS_filename_list.find(delim, start);
			depthmap_filename_list.push_back(FLAGS_filename_list.substr(start, end - start));
		}
		if (depthmap_filename_list.size() < 2)
			LOG(ERROR) << "The input depth maps should be at least 2 depth maps.";
		depthmap_stitch(root_dir, depthmap_filename_list, FLAGS_method);
	}
	else if (FLAGS_type.compare("debug") == 0)
		//create_debug_data(root_dir);
		create_debug_data_multi(root_dir);
}