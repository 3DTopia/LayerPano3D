#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <vector>
#include <map>


/**
 * Create and store the depth map data.
 * There are composed with two depth maps, the first one is reference image, the second is the target image.
 * Reference_Image = Target_Image * scale_coeff + offset_coeff.
 */
class DataImitator
{
public:
	DataImitator();

	~DataImitator();

	/**
	 * Create a depth map pair for debug.
	 * target depth map = reference depth map * scale + offset.
	 */
	std::vector<cv::Mat>  make_depthmap_pair_simple();
	std::vector<cv::Mat>  make_depthmap_pair_random();

	/**
	 * Make mock pixel corresponding which is from target to reference, and output to JSON file .
	 * The first depth on the top and second on the bottom. So the 1st map bottom overlap the 2nd map's top.
	 */
	std::map<int, std::map<int, cv::Mat>> make_corresponding_json();
	void make_pixel_corresponding();

	/**
	 * Create the depth maps alignment coefficients.
	 */
	std::vector<cv::Mat> make_aligncoeffs_simple();
	std::vector<cv::Mat> make_aligncoeffs_random();

	/**
	 * Save depth maps and coefficients to files.
	 */
	void output_date();

	// coefficients *.json file path.
	std::string json_filepath;

	// the output file path
	std::string depthmap_ref_filepath;
	std::string depthmap_tar_filepath;

	std::string corr_ref2tar_filepath;
	std::string corr_tar2ref_filepath;

	// depth map information
	int depthmap_width = 50;
	int depthmap_hight = 80;

	// the coefficient grid size
	int coeff_grid_width = 5;
	int coeff_grid_height = 10;

	// how much area overlap
	float depthmap_overlap_ratio = 1.0;

	// depth map scale and offset
	float depth_scale = 0.2;
	float depth_offset = 6.0;

private:

	// The depth map deform coefficients. scale & offset array
	std::vector<cv::Mat> coeff_list;

	// The deformed depth map.
	std::vector<cv::Mat> depthmap_list;

	// the images corresponding relationship
	cv::Mat corresponding_mat_ref2tar;
	cv::Mat corresponding_mat_tar2ref;
};

/** Report mock data information parameters */
std::ostream& operator <<(std::ostream& os, const DataImitator& di);

// Generate the three image sequence.
// 1st image is reference image, 1st = 2nd * scale + offset, 2nd = 3rd * scale + offset
class DataImitatorMultiImage
{

public:
	// the mock depth map frame number
	// the frame index range is [0, frame_number - 1]
	const int frame_number = 3;

	// depth map information
	int depthmap_width = 60;
	int depthmap_hight = 80;

	// the coefficient grid size
	int coeff_grid_width = 6;
	int coeff_grid_height = 8;

	// how much area overlap, the depth map horizontally overlaps.
	float depthmap_overlap_ratio = 0.4;

	// depth map scale and offset
	float depth_scale = 0.2;
	float depth_offset = 6.0;

	// the root folder to output test data
	std::string output_root_dir;

	// create the template of depth map and coefficient 
	void initial();

	/**
	 * Generate the depth maps alignment coefficients.
	 */
	void make_aligncoeffs_simple(); // constant value number
	void make_aligncoeffs_random(); // random number

	// the deform grid weight is 
	void make_depthmap_pair(); // constant value number

	/**
	 * Make mock pixel corresponding which is from target to reference, and output to JSON file .
	 * The first depth on the top and second on the bottom. So the 1st map bottom overlap the 2nd map's top.
	 */
	void make_pixel_corresponding();

	/**
	 * Save depth maps and coefficients to files.
	 */
	void output_date();

	void report_data_parameters();

public:

	// 1st image index is 0, 2nd is 1, 3rd is 2.
	std::map<int, std::map<int, cv::Mat>> pixel_corresponding;

	// the deformation coefficient 3rd -> 2nd, 2nd -> 1st
	std::map<int, std::map<int, cv::Mat>> coeff_scale_list;
	std::map<int, std::map<int, cv::Mat>> coeff_offset_list;

	// the depth map list 1st, 2nd, 3rd, the last depth map is the reference depth map
	std::map<int, cv::Mat> depthmap_list; // 
	std::map<int, cv::Mat> depthmap_template_list;

	// the template of the depth map & alignment coefficient
	cv::Mat depthmap_ref_template;
	double scale_mean = 0;
	double scale_stddev = 5;
	double offset_mean = 0;
	double offset_stddev = 80.0;
	std::map<int, std::map<int, cv::Mat>> coeff_template_scale;
	std::map<int, std::map<int, cv::Mat>> coeff_template_offset;

	// output file name expression
	// pixels corresponding filename regular expression
	std::string depthmap_filename_exp = "img0_depth_$subimageindex.pfm";
	std::string pixelcorr_filename_exp = "img0_corr_$srcindex_$tarindex.json";
	std::string aligncoeff_filename_exp = "img0_coeff_align.json";

};
