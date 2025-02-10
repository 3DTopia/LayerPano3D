#include "include/data_io.hpp"
#include "include/depthmap_utility.hpp"
#include "include/depthmap_stitcher.hpp"
#include "include/timer.hpp"

#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <random>
#include <regex>
#include <iterator>

//const std::string test_data_dir("D:/workspace_windows/InstaOmniDepth/data/");
const std::string test_data_dir("D:/workspace_windows/InstaOmniDepth/InstaOmniDepth_github/code/cpp/bin/Release/");
const std::string data_root_dir = test_data_dir + "fisheye_00/";

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

TEST(DepthmaputilTest, defbilinear_weightorm)
{
	const int grid_width = 5;
	const int grid_height = 8;
	const int image_width = 50;
	const int image_height = 80;
	double *weight_list = (double *)malloc(grid_width * grid_height * sizeof(double));

	for (int y = 0; y < image_height; y = y + 10)
	{
		for (int x = 0; x < image_width; x = x + 10)
		{
			DepthmapUtil::bilinear_weight(weight_list,
										  image_width, image_height,
										  grid_width, grid_height,
										  x, y);
			cv::Mat bilinear_weight_list_mat(cv::Size(grid_width, grid_height), CV_64FC1, weight_list);
		}
	}
}

//void test_depthmap_deform()
TEST(DepthmaputilTest, deform)
{
	// 0) create depth map
	int image_width = 50;
	int image_height = 80;

	cv::Mat depthmap = cv::Mat::ones(image_height, image_width, CV_64FC1);
	int counter = 0;
	for (int row_index = 0; row_index < depthmap.rows; row_index++)
	{
		for (int col_index = 0; col_index < depthmap.cols; col_index++)
		{
			depthmap.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}

	// 1) create offset and scale grid
	int grid_width = 5;
	int grid_height = 8;
	cv::Mat scale_mat = cv::Mat::ones(grid_height, grid_width, CV_64FC1);
	counter = 0;
	for (int row_index = 0; row_index < scale_mat.rows; row_index++)
	{
		for (int col_index = 0; col_index < scale_mat.cols; col_index++)
		{
			scale_mat.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}
	cv::Mat offset_mat = cv::Mat::ones(grid_height, grid_width, CV_64FC1);
	counter = 0;
	for (int row_index = 0; row_index < offset_mat.rows; row_index++)
	{
		for (int col_index = 0; col_index < offset_mat.cols; col_index++)
		{
			offset_mat.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}

	//cv::randu(scale_mat, cv::Scalar(-10), cv::Scalar(10));
	//cv::randu(offset_mat, cv::Scalar(-10), cv::Scalar(10));

	cv::Mat depthmap_deformed;
	DepthmapUtil::deform(depthmap, depthmap_deformed, scale_mat, offset_mat);
}

TEST(DepthmaputilTest, deform_load_data)
{
	// 0) create depth map
	int image_width = 50;
	int image_height = 80;
	cv::Mat depthmap = cv::Mat::ones(image_height, image_width, CV_64FC1);

	//cv::Mat depthmap;
	//DepthmapIO::load(test_data_dir + "img0_depth_001.pfm", depthmap);

	// 1) load offset and scale grid from file
	std::string coeff_file_path = test_data_dir + "img0_coeff.json";
	AlignCoeff align_coeff;
	align_coeff.initial(5, 8, 2);
	align_coeff.load(coeff_file_path);
	//cv::Mat scale_mat = align_coeff.coeff_scale_mat[1];
	cv::Mat scale_mat = cv::Mat::ones(align_coeff.coeff_scale_mat[1].size(), CV_64FC1);
	cv::Mat offset_mat = align_coeff.coeff_offset_mat[1];
	//cv::Mat offset_mat = cv::Mat::ones(align_coeff.coeff_offset_mat[1].size(), CV_64FC1);

	cv::Mat depthmap_deformed;
	DepthmapUtil::deform(depthmap, depthmap_deformed, scale_mat, offset_mat);
}

TEST(DepthmaputilTest, normalization)
{
	cv::Mat depthmap = cv::Mat::ones(50, 80, CV_64FC1);
	//cv::randu(depthmap, cv::Scalar(-10), cv::Scalar(10));
	int counter = 0;
	for (int row_index = 0; row_index < depthmap.rows; row_index++)
	{
		for (int col_index = 0; col_index < depthmap.cols; col_index++)
		{
			depthmap.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}

	double mean, stddev;
	cv::Mat depthmap_norm = DepthmapUtil::depthmap_orig2norm(depthmap, mean, stddev);
	cv::Mat depthmap_restore = DepthmapUtil::depthmap_norm2orig(depthmap_norm, mean, stddev);

	cv::Mat diff = depthmap - depthmap_restore;
}

TEST(DepthmapStitcherTest, subpixel_performance)
{
	// create test data
	unsigned int image_width = 200;
	unsigned int image_height = 150;
	cv::Mat img = cv::Mat::zeros(image_height, image_width, CV_64FC1);

	int counter = 0;
	for (int row_idx = 0; row_idx < image_height; row_idx++)
		for (int col_idx = 0; col_idx < image_width; col_idx++)
		{
			img.at<double>(row_idx, col_idx) = counter;
			counter++;
		}

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> width_dist(0, image_width - 1);
	std::uniform_int_distribution<std::mt19937::result_type> height_dist(0, image_height - 1);
	cv::Point2f pt(width_dist(rng), height_dist(rng));

	Timer timer;
	int test_time = 100000;

	// method 1:
	double data_method_1;
	timer.start();
	for (int i = 0; i < test_time; i++)
	{
		cv::Mat patch;
		cv::Mat imgF;
		img.convertTo(imgF, CV_32FC1);
		cv::getRectSubPix(imgF, cv::Size(1, 1), pt, patch);
		data_method_1 = patch.at<float>(0, 0);
	}
	printf("Method 1: %f, %f ms\n", data_method_1, timer.duration_ms());
	timer.stop();

	// method 2:
	timer.start();
	double data_method_2;
	for (int i = 0; i < test_time; i++)
	{
		cv::Mat patch;
		cv::remap(img, patch, cv::Mat(1, 1, CV_32FC2, &pt), cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
		data_method_2 = patch.at<double>(0, 0);
	}
	printf("Method 2: %f, %f ms\n", data_method_2, timer.duration_ms());
	timer.stop();

	// method 3:
	assert(!img.empty());
	assert(img.channels() == 3);
	timer.start();
	double data_method_3;
	for (int i = 0; i < test_time; i++)
	{
		int x = (int)pt.x;
		int y = (int)pt.y;
		int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
		int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
		int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
		int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

		float a = pt.x - (float)x;
		float c = pt.y - (float)y;

		data_method_3 = (double)cvRound(
			(img.at<double>(y0, x0) * (1.f - a) + img.at<double>(y0, x1) * a) * (1.f - c) +
			(img.at<double>(y1, x0) * (1.f - a) + img.at<double>(y1, x1) * a) * c);
	}
	printf("Method 3: %f , %f ms\n", data_method_3, timer.duration_ms());
	timer.stop();
}

void test_data_loading()
{
	// 0) load pixels corresponding from JSON
	std::string pixles_corresponding_json = data_root_dir + "pixels_corresponding_0_7.json";
	cv::Mat pixles_corresponding;
	std::string src_filename;
	std::string tar_filename;
	PixelsCorrIO::load(pixles_corresponding_json, src_filename, tar_filename, pixles_corresponding);

	// 1) load depth from .pfm file.
	std::string pfm_depthmap_filepath = data_root_dir + "img0_000.pfm";
	cv::Mat depth_mat;
	DepthmapIO::load(pfm_depthmap_filepath, depth_mat);
}
