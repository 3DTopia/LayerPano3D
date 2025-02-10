#include "depthmap_utility.hpp"


void DepthmapUtil::normalize(const cv::Mat& depthmap, cv::Mat& depthmap_normalized)
{
	// 1) compute the media number
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&depthmap_normalized, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	int bin = 0;
	double median = -1.0;
	double middle_idx = (depthmap_normalized.rows * depthmap_normalized.cols) / 2;
	for (int i = 0; i < histSize && median < 0.0; ++i)
	{
		bin += cvRound(hist.at< float >(i));
		if (bin > middle_idx && median < 0.0)
			median = i;
	}

	//// normalize the image
	//dev_dispmap = np.sum(np.abs(dispmap[mask] - median_dispmap)) / np.sum(mask)
	//dispmap_norm = np.full(dispmap.shape, np.nan, dtype = np.float64)
	//dispmap_norm[mask] = (dispmap[mask] - median_dispmap) / dev_dispmap

}


void DepthmapUtil::deform(const cv::Mat& depth_map_original, cv::Mat& depthmap_deformed, const cv::Mat& scale_mat, const cv::Mat& offset_mat)
{
	int image_width = depth_map_original.cols;
	int image_height = depth_map_original.rows;

	int grid_width = scale_mat.cols;
	int grid_height = scale_mat.rows;

	// 1) compute the bilinear weight for the scale and offset
	// 1-0) get the mesh grid
	// the sub-pixels location used to interpolation the scale and offset grid
	std::vector<float> grid_list_x, grid_list_y;
	float grid_col_interval = (image_width - 1.0) / (grid_width - 1.0);
	for (int i = 0; i < image_width; i++)
		grid_list_x.push_back((float)i / grid_col_interval);
	cv::Mat grid_list_x_mat = cv::Mat(grid_list_x);

	float grid_row_interval = (image_height - 1.0) / (grid_height - 1.0);
	for (int i = 0; i < image_height; i++)
		grid_list_y.push_back((float)i / grid_row_interval);
	cv::Mat  grid_list_y_mat = cv::Mat(grid_list_y);

	cv::Mat meshgrid_x, meshgrid_y;
	cv::repeat(grid_list_x_mat.reshape(1, 1), image_height, 1, meshgrid_x);
	cv::repeat(grid_list_y_mat.reshape(1, image_height), 1, image_width, meshgrid_y);

	// 1-1) compute the pixels scale and offset
	// the origin is Top-Left
	cv::Mat scale_mat_pixels, offset_mat_pixels;
	cv::remap(scale_mat, scale_mat_pixels, meshgrid_x, meshgrid_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::remap(offset_mat, offset_mat_pixels, meshgrid_x, meshgrid_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

	// 2) compute the depth map
	depthmap_deformed = scale_mat_pixels.mul(depth_map_original) + offset_mat_pixels;
}


void DepthmapUtil::bilinear_weight(double* weight_list,
	const int image_width, const int image_height,
	const int grid_width, const int grid_height,
	const double x, const double y)
{
	std::memset(weight_list, 0, grid_height * grid_width * sizeof(double));
	// for (int idx = 0; idx < grid_height * grid_width; idx++)
	// {
	// 	weight_list[idx] = 0;
	// }

	// get the range of the pixel's block, and the origin is Top-Left
	// the col up and low
	double grid_col_interval = (image_width - 1.0) / (grid_width - 1.0);
	int grid_col_low = int(x / grid_col_interval);
	double grid_col_low_pixel = grid_col_low * grid_col_interval;
	int grid_col_up;
	if (grid_col_low_pixel == x)
		grid_col_up = grid_col_low;
	else
		grid_col_up = grid_col_low + 1;
	double grid_col_up_pixel = grid_col_up * grid_col_interval;

	// the row up and low
	double grid_row_interval = (image_height - 1.0) / (grid_height - 1.0);
	int grid_row_low = int(y / grid_row_interval);
	double grid_row_low_pixel = grid_row_low * grid_row_interval;
	int grid_row_up;
	if (grid_row_low_pixel == y)
		grid_row_up = grid_row_low;
	else
		grid_row_up = grid_row_low + 1;
	double grid_row_up_pixel = grid_row_up * grid_row_interval;

	// compute the bilinear weights
	if (grid_row_low == grid_row_up && grid_col_low != grid_col_up)
	{ 
		// in the same row
		double weight_denominator = grid_col_up_pixel - grid_col_low_pixel;
		weight_list[grid_row_low * grid_width + grid_col_low] = (grid_col_up_pixel - x) / weight_denominator; //the Top-Left && Bottom-Left
		weight_list[grid_row_low * grid_width + grid_col_up] = (x - grid_col_low_pixel) / weight_denominator; // The Top-Right && Bottom-Right
	}
	else if (grid_row_low != grid_row_up && grid_col_low == grid_col_up)
	{
		// in the same column
		double weight_denominator = grid_row_up_pixel - grid_row_low_pixel;
		weight_list[grid_row_low * grid_width + grid_col_low] = (grid_row_up_pixel - y) / weight_denominator; //the Top-Left && Top-Right
		weight_list[grid_row_up * grid_width + grid_col_low] = (y - grid_row_low_pixel) / weight_denominator; // The Bottom-Left && Bottom-Right
	}
	else if (grid_row_low == grid_row_up && grid_col_low == grid_col_up)
	{
		// in the same point
		weight_list[grid_row_low * grid_width + grid_col_low] = 1.0;
	}
	else
	{
		double weight_denominator = (grid_row_up_pixel - grid_row_low_pixel) * (grid_col_up_pixel - grid_col_low_pixel);
		weight_list[grid_row_low * grid_width + grid_col_low] = (grid_col_up_pixel - x) * (grid_row_up_pixel - y) / weight_denominator; //the Top-Left
		weight_list[grid_row_low * grid_width + grid_col_up] = (x - grid_col_low_pixel) * (grid_row_up_pixel - y) / weight_denominator; // The Top-Right
		weight_list[grid_row_up * grid_width + grid_col_low] = (grid_col_up_pixel - x) * (y - grid_row_low_pixel) / weight_denominator; // The Bottom-Left
		weight_list[grid_row_up * grid_width + grid_col_up] = (x - grid_col_low_pixel) * (y - grid_row_low_pixel) / weight_denominator; // The Bottom-Right
	}
}


cv::Mat DepthmapUtil::depthmap_orig2norm(const cv::Mat& orig_mat, double& mean, double& stddev)
{
	cv::Scalar mean_, stddev_;
	cv::meanStdDev(orig_mat, mean_, stddev_);
	mean = mean_[0];
	stddev = stddev_[0];

	cv::Mat norm_mat = (orig_mat - mean_) / stddev_;
	return norm_mat;
}


cv::Mat DepthmapUtil::depthmap_norm2orig(const cv::Mat& norm_mat, const double mean, const double dev)
{
	return norm_mat * dev + mean;
}
