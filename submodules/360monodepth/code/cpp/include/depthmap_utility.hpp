#pragma once

#include <opencv2/opencv.hpp>

class DepthmapUtil
{

public:
	static void normalize(const cv::Mat& depthmap, cv::Mat& depthmap_normalized);

	static void deform(const cv::Mat& depthmap, cv::Mat& depthmap_deformed, const cv::Mat& scale_mat, const cv::Mat& offset_mat);

	static void bilinear_weight(double* weight_list,
		const int image_width, const int image_height,
		const int grid_width, const int grid_height,
		const double x, const double y);

	// normalized the depth map
	static cv::Mat depthmap_orig2norm(const cv::Mat& orig_mat, double& mean, double& dev);
	static cv::Mat depthmap_norm2orig(const cv::Mat& norm_mat, const double mean, const double dev);
};
