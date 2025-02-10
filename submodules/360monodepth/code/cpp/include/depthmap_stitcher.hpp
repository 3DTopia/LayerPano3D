#pragma once

#include <vector>
#include <map>
#include <string>

#include <opencv2/opencv.hpp>

class AlignCoeff
{

public:
	void initial(const int grid_width, const int grid_height, const int depthmap_number);

	void set_value_const(const float scale = 0.1, const float offset = 0.0);

	void set_value_mat(const std::vector<cv::Mat>& coeff_scale, const std::vector<cv::Mat>& coeff_offset);

	/**  use random parameter as the initial parameters. */
	void set_value_rand();

	void save(const std::string& file_path, const std::vector<std::string>& filename_list);
	void load(const std::string& file_path);

	// use to parser
	int coeff_rows; // gird height, row number
	int coeff_cols; // grid width, column number
	int coeff_number; // the coefficients (scale & offset) pairs number

	// share the same memory with coeff_scale and coeff_offset
	std::vector<cv::Mat> coeff_scale_mat;
	std::vector<cv::Mat> coeff_offset_mat;

	// a continue memory store coefficient for all image, row major
	std::shared_ptr<double[]> coeff_scale;
	std::shared_ptr<double[]> coeff_offset;
};


/** Output the coefficients parameters */
std::ostream& operator <<(std::ostream& os, const AlignCoeff& di);

class DepthmapStitcher
{

public:
	DepthmapStitcher();

	~DepthmapStitcher();

	/**
	 * Load depth map and pixels corresponding.
	 * Create the depth map internal index, by get the file index from filename and re-assign index based on filename sequence.
	 * The file naming convention please refere @see doc/readme.md.
	 *
	 * @param data_root_dir the root folder of image data.
	 * @param depthmap_filename_list the sub-depth map's file name.
	 */
	void load_data(const std::string& data_root_dir, const std::vector<std::string> depthmap_filename_list);

	/**
	 * Assign depth map and pixels corresponding, and generate proxy file name.
	 * Create the depth internal index, by get the file index from filename and re-assign index based on filename sequence.
	 * The file naming convention please refere @see doc/readme.md.
	 *
	 * @param data_root_dir the root folder of image data.
	 * @param depthmap_original_data the sub-depth map's file name.
	 * @param depthmap_original_ico_index the sub-depth icosahedron face index.
	 * @param pixels_corresponding_list_data the pixel corresponding relation ship between two image.
	 */
	void initial_data(const std::string& data_root_dir,
		const std::vector<cv::Mat>& depthmap_original_data,
		const std::vector<int>& depthmap_original_ico_index,
		const std::map<int, std::map<int, cv::Mat>>& pixels_corresponding_list_data);

	/**
	 * Get the grid size .
	 * The sub-class should call this super-call function.
	 */
	virtual void initial(const int grid_width, const int grid_height);

	void set_coeff(const std::vector<cv::Mat>& coeff_scale,
		const std::vector<cv::Mat>& coeff_offset);

	/**
	 * Compute each depth maps S and A relative reference depth map.
	 * Use the first depth map in depth_map_list as the reference map.
	 * And enumerate all image pairs to compute the scale and offset.
	 */
	virtual void compute_align_coeff() = 0;

	/**
	 * Save and load coeff to and from file.
	 */
	void get_align_coeff(std::vector<cv::Mat>& coeff); 
	void save_align_coeff();

	/**
	 * Use align coefficients parameters to adjust the depth map.
	 */
	void align_depthmap_all();
	void align_depthmap(const int depthmap_intidx);

	/**
	 * Return the aligned depth maps.
	 */
	std::vector<cv::Mat> get_aligned_depthmap() { return depthmap_aligned; }

	/**
	 * save aligned depth map to data root folder.
	 */
	void save_aligned_depthmap();

	/**
	 * Evaluate the error between the original and new depth maps.
	 * Warp the adjusted depth map to others depth map with the corresponding relationship, and compute the error.
	 */
	void report_error();

	float getColorSubpix(const cv::Mat& img, const cv::Point2f pt);

	// depth map's sub map filename regular expression
	int index_digit_number = 3;
	std::string depthmap_filename_regexp = "[a-zA-Z0-9\\_]*\\_disp\\_erp\\_[0-9]{3}.pfm";
	std::string rgb_filename_regexp = "[a-zA-Z0-9\\_]*\\_rgb\\_[0-9]{3}.jpg";
	std::string corr_filename_regexp = "[a-zA-Z0-9\\_]*\\_corr\\_[0-9]{3}\\_[0-9]{3}.json";
	// pixels corresponding filename regular expression
	std::string pixelcorr_filename_exp = "$prefix_corr_$srcindex_$tarindex.json";

	// proxy filename for data fetch from memory
	std::string proxy_filename_depthmap_exp = "depthmap_depth_$index.pymodule";
	//std::string proxy_filename_pixelcorr_exp = "depthmap_corr_$index.json";s

	// output file name expression
	std::string depthmap_aligned_filename_exp = "$prefix_depth_$subindex_aligned.pfm";
	std::string align_coeff_filename_exp = "$prefix_coeff_align.json";

	// cost function terms weight
	float weight_reprojection = 1.0; // re-projection term lambda
	float weight_smooth = 10e-4;     // smooth term lambda
	float weight_scale = 10e-2;      // scale term lambda

	// the reference depth map index, start from 0. 0 is the first depth map index.
	int depthmap_ref_extidx = 0;

	// Ceres solver opinions
	int ceres_num_threads = -1;
	int ceres_max_num_iterations = -1;
	int ceres_max_linear_solver_iterations = -1;
	int ceres_min_linear_solver_iterations = -1;

	// perpixel or pergrid term weight 
	bool projection_per_pixelcost_enable = false;
	bool smooth_pergrid_enable = false;

    // the scalar and offset for each depth map, CV_64FC1 for each depth map
    // std::vector<cv::Mat> scale_list;
    // std::vector<cv::Mat> offset_list;
    AlignCoeff coeff_so;

protected:
	/**
	 * Compute the bilinear interpolation weight int the sparse weight grid.
	 * @param weight_list the offset or scale weight list. It is row-major.
	 * @param image_width the depth width.
	 */
	void get_bilinear_weight(double* weight_list,
		const int image_width, const int image_height,
		const int grid_row_number, const int grid_col_number,
		const double x, const double y);

	// the container of the depth image.
	std::vector<cv::Mat> depthmap_original;

	// the aligned depth map
	std::vector<cv::Mat> depthmap_aligned;

	// the mapping form the internal depth image index to depth file name.
	std::vector<std::string> filename_list;

	// the pixels corresponding between two image, it's row major. the sub-image index is internal index.
	//  [cur_y, cur_x, ref_y, ref_x].
	// The pixels_corresponding_list[3][2] is from 3 to 2. 2 is reference depth image.
	// Note the index are re-assigned based on filename sequence.
	std::map<int, std::map<int, cv::Mat>> pixels_corresponding_list;

	// depth map and corr file folder path
	std::string data_root_dir;
	// the prefix for depth map, coefficients and corresponding
	std::string filename_prefix;

	// the map between internal index and external index
	// external index is the icosahedron sub-image index [0-19], internal index used by Ceres solver.
	std::map<int, int> extidx2intidx;
	std::map<int, int> intidx2extidx;

	// The scale and offset grid size, @see hedman2018instant
	int grid_width = 5; // the grid column number
	int grid_height = 5; // the grid row number
};