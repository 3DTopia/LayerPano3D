#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <map>

// Note: all cv::Mat data depth is double (64F).
/**
 * Initial the depth map alignment module run time environment.
 * 
* @param method The depth map stitcher method name.
 * 
 * @return int is < 0 if there is error.
 */
int init(const std::string &method);

/**
 * Clear the depth map alignment run time environment.
 * 
 * @return int 
 */
int shutdown();

/**
 * Report the error between subimages.
 * 
 * @return int 
 */
int report_aligned_depthmap_error();

/**
 * Create mock data for debug. 
 * @param depthmap The synthetic depth map
 * @param align_coeff The synthetic depth map deformation alignment coefficients.
 * @param pixels_corresponding_list The synthetic depth map pixel corresponding relationship.
 * @param debug_data_type The debug data type, 0 is the simple debug data, 1 is the random debug data.
 * @param frame_number The frame number of the generated debug data.
 */
void create_debug_data(std::vector<cv::Mat> &depthmap,
					   std::vector<cv::Mat> &align_coeff,
					   std::map<int, std::map<int, cv::Mat>> &pixels_corresponding_list,
					   const int debug_data_type,
					   const int frame_number);

/**
* Set the Ceres solver options. Set it to a number less than or equal to 0 to use default value.
 * @param num_threads:
 * @param max_num_iterations:
 * @param max_linear_solver_iterations:
 * @param min_linear_solver_iterations:
 */
int solver_params(const int num_threads,
				  const int max_num_iterations,
				  const int max_linear_solver_iterations,
				  const int min_linear_solver_iterations);

/**
 * Align sub-images depth maps.
 * @param root_dir if is not empty, output aligned depth map and coefficients this folder.
 * @param terms_weight The term wight, projection term, smooth term and regulation term.
 * @param depthmap_original Original depth map.
 * @param depthmap_original_ico_index The original depth maps icosahedron index.
 * @param reference_depthamp_ico_index The reference depth map index of the icosahedron.
 * @param pixels_corresponding_list The pixel corresponding relationship between difference sub-images.
 * @param align_coeff_grid_height the align coefficients gird height, if use default set it to -1.
 * @param align_coeff_grid_width the align coefficients gird width, if use default set it to -1.
 * @param reproj_perpixel_enable the energy term use perpixel cost.
 * @param smooth_pergrid_enable the smooth term use pergrid cost.
 * @param align_coeff_initial_scale the initial scale coefficient
 * @param align_coeff_initial_offset the initial offset coefficient.
 * @param log_level the glog output level, 0 is Warning, 1 is Info
 * @param depthmap_aligned Aligned depth maps.
 * @param align_coeff The depth align coefficients.
 */
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
	std::vector<cv::Mat> &align_coeff);