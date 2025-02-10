#include "python_binding.hpp"

#include "depthmap_stitcher.hpp"
#include "depthmap_stitcher_enum.hpp"
#include "depthmap_stitcher_group.hpp"
#include "data_imitator.hpp"

#include <glog/logging.h>

#include <string>
#include <iostream>
#include <algorithm>

std::shared_ptr<DepthmapStitcher> depthmap_stitcher;

// create fake data for debug
void create_debug_data(std::vector<cv::Mat> &depthmap,
					   std::vector<cv::Mat> &align_coeff,
					   std::map<int, std::map<int, cv::Mat>> &pixels_corresponding_list,
					   const int debug_data_type,
					   const int frame_number)
{
	std::stringstream ss;
	ss << "Create synthetic debug data :";
	if (frame_number == 2)
	{
		ss << " Frame Number is: " << frame_number;
		DataImitator di;
		di.depthmap_ref_filepath.clear();
		di.depthmap_tar_filepath.clear();
		di.corr_ref2tar_filepath.clear();
		di.corr_tar2ref_filepath.clear();
		//di.depthmap_overlap_ratio = 0.5;
		di.depthmap_overlap_ratio = 1.0;
		di.depth_scale = 0.2;
		di.depth_offset = 6.0;

		if (debug_data_type == 0)
		{
			ss << "Deform coefficient is Simple.";
			LOG(INFO) << ss.str();
			align_coeff = di.make_aligncoeffs_simple();
			depthmap = di.make_depthmap_pair_simple();
		}
		else if (debug_data_type == 1)
		{
			ss << "Deform coefficient is Random.";
			LOG(INFO) << ss.str();
			align_coeff = di.make_aligncoeffs_random();
			depthmap = di.make_depthmap_pair_random();
		}
		else
		{
			LOG(ERROR) << "Only support debug data type is simple and random.";
		}
		pixels_corresponding_list = di.make_corresponding_json();
	}
	else if (frame_number > 2)
	{
		// TODO create multi frame for depth map alignment
		if (debug_data_type == 0)
		{
			ss << "Deform coefficient is Simple.";
			LOG(INFO) << ss.str();
		}
		else if (debug_data_type == 1)
		{
			ss << "Deform coefficient is Random.";
			LOG(INFO) << ss.str();
		}
		else
		{
			LOG(ERROR) << "Only support debug data type is simple and random.";
		}
	}
	else
	{
		LOG(ERROR) << "Debug data frame number is " << frame_number;
	}
}

int solver_params(const int num_threads,
				  const int max_num_iterations,
				  const int max_linear_solver_iterations,
				  const int min_linear_solver_iterations)
{
	int py_ceres_num_threads = -1;
	int py_max_num_iterations = -1;
	int py_max_linear_solver_iterations = -1;
	int py_min_linear_solver_iterations = -1;

	py_ceres_num_threads = num_threads;
	py_max_num_iterations = max_num_iterations;
	py_max_linear_solver_iterations = max_linear_solver_iterations;
	py_min_linear_solver_iterations = min_linear_solver_iterations;
	//

	// set Ceres soler options
	if (py_ceres_num_threads > 0)
	{
		LOG(INFO) << "The Ceres thread number is :" << py_ceres_num_threads;
		depthmap_stitcher->ceres_num_threads = py_ceres_num_threads;
		py_ceres_num_threads = -1;
	}
	if (py_max_num_iterations > 0)
	{
		LOG(INFO) << "The Ceres max iteration number is :" << py_max_num_iterations;
		depthmap_stitcher->ceres_max_num_iterations = py_max_num_iterations;
		py_max_num_iterations = -1;
	}
	if (py_max_linear_solver_iterations > 0)
	{
		LOG(INFO) << "The Ceres max linear solver iteration number is :" << py_max_linear_solver_iterations;
		depthmap_stitcher->ceres_max_linear_solver_iterations = py_max_linear_solver_iterations;
		py_max_linear_solver_iterations = -1;
	}
	if (py_min_linear_solver_iterations > 0)
	{
		LOG(INFO) << "The Ceres min linear solver iteration number is :" << py_min_linear_solver_iterations;
		depthmap_stitcher->ceres_min_linear_solver_iterations = py_min_linear_solver_iterations;
		py_min_linear_solver_iterations = -1;
	}

	return 1;
}

int init(const std::string &method)
{
	// initial log
	static bool glog_initialized = false;
	const int log_level = 1;
	if (log_level == 0)
		FLAGS_stderrthreshold = google::GLOG_WARNING;
	else if (log_level == 1)
		FLAGS_stderrthreshold = google::GLOG_INFO;

    if (!glog_initialized)
    {
    	google::InitGoogleLogging("depthmap_stitch");
    	glog_initialized = true;
    }

	// initial the depth map alignment stitcher
	if (method.compare("enum") == 0)
	{
		depthmap_stitcher = std::make_shared<DepthmapStitcherEnum>();
		LOG(INFO) << "Depth map align with enum method.";
	}
	else if (method.compare("group") == 0)
	{
		depthmap_stitcher = std::make_shared<DepthmapStitcherGroup>();
		LOG(INFO) << "Depth map align with group method.";
	}
	else
	{
		LOG(ERROR) << "The specified method name " << method << " is wrong.";
		return -1;
	}
	return 0;
}

int shutdown()
{
	// shut down glog
	google::ShutdownGoogleLogging();
	return 0;
}

// align sub-image depth-map
int report_aligned_depthmap_error()
{
	// aligned depth map error report
	depthmap_stitcher->report_error();
	return 0;
}

// stitch depth maps
int depthmap_stitch(
	const std::string &root_dir,
	const std::vector<float> &terms_weight,
	const std::vector<cv::Mat> &depthmap_original,
	const std::vector<int> &depthmap_original_ico_index,
	const int reference_depthamp_ico_index,
	const std::map<int, std::map<int, cv::Mat>> &pixels_corresponding_list,
	const int align_coeff_grid_height,
	const int align_coeff_grid_width,
	const bool reproj_perpixel_enable,
	const bool smooth_pergrid_enable,
	const std::vector<cv::Mat> &align_coeff_initial_scale,
	const std::vector<cv::Mat> &align_coeff_initial_offset,
	std::vector<cv::Mat> &depthmap_aligned,
	std::vector<cv::Mat> &align_coeff)
{
	// check the term's weight
	if (terms_weight.size() != 3)
		return -1;

	// check the reference depth map
	if (reference_depthamp_ico_index >= 0)
	{
		std::vector<int>::const_iterator it = std::find(depthmap_original_ico_index.begin(), depthmap_original_ico_index.end(), reference_depthamp_ico_index);
		if (it != depthmap_original_ico_index.end())
			LOG(ERROR) << "The reference index " << reference_depthamp_ico_index << " is not in the ico depth map list.";
	}

	depthmap_stitcher->depthmap_ref_extidx = reference_depthamp_ico_index;

	// set term's weight
	depthmap_stitcher->weight_reprojection = terms_weight[0]; // re-projection term lambda
	depthmap_stitcher->weight_smooth = terms_weight[1];		  // smooth term lambda
	depthmap_stitcher->weight_scale = terms_weight[2];		  // scale term lambda
	LOG(INFO) << "Terms weights of re-projection:" << depthmap_stitcher->weight_reprojection << ",\tsmooth:" << depthmap_stitcher->weight_smooth << ",\tscale:" << depthmap_stitcher->weight_scale;

	// set if use the per-pixel/grid weight 
	depthmap_stitcher->projection_per_pixelcost_enable = reproj_perpixel_enable;
	depthmap_stitcher->smooth_pergrid_enable = smooth_pergrid_enable;
	LOG(INFO) << "projection_per_pixelcost_enable : " << reproj_perpixel_enable;
	LOG(INFO) << "smooth_pergrid_enable : " << smooth_pergrid_enable;	

	// convert external data to internal data format
	depthmap_stitcher->initial_data(root_dir, depthmap_original, depthmap_original_ico_index, pixels_corresponding_list);

	// set the coefficients grid size
	if (align_coeff_grid_height != -1 && align_coeff_grid_width != -1)
	{
		depthmap_stitcher->initial(align_coeff_grid_width, align_coeff_grid_height);
		LOG(INFO) << "Set the coefficient grid size to " << align_coeff_grid_height << " x " << align_coeff_grid_width;
	}
	else
		LOG(ERROR) << "Set the coefficient grid size do not set!";

	// set the alignment coefficient
	if (!align_coeff_initial_scale.empty() && !align_coeff_initial_offset.empty())
	{
		depthmap_stitcher->set_coeff(align_coeff_initial_scale, align_coeff_initial_offset);
	}

	LOG(INFO) << "Begin compute alignment coefficient.";
	depthmap_stitcher->compute_align_coeff();
	depthmap_stitcher->align_depthmap_all();
	if (!root_dir.empty())
	{
		depthmap_stitcher->save_aligned_depthmap();
		depthmap_stitcher->save_align_coeff();
	}

	// return alignment data and coefficients.
	depthmap_aligned = depthmap_stitcher->get_aligned_depthmap();
	depthmap_stitcher->get_align_coeff(align_coeff);

	depthmap_stitcher->coeff_so.coeff_scale_mat.clear();
	depthmap_stitcher->coeff_so.coeff_offset_mat.clear();

	return 0;
}