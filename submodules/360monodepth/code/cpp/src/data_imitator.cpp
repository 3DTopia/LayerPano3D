#include "data_imitator.hpp"
#include "pfm_io.hpp"
#include "data_io.hpp"
#include "depthmap_utility.hpp"
#include "depthmap_stitcher.hpp"

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include <string>
#include <iostream>
#include <regex>
#include <filesystem>
namespace fs = std::filesystem;

DataImitator::DataImitator()
{

}

DataImitator::~DataImitator()
{

}

std::vector<cv::Mat> DataImitator::make_depthmap_pair_simple()
{
	// 0) create depth map
	cv::Mat depthmap_tar_data = cv::Mat::ones(depthmap_hight, depthmap_width, CV_64FC1);
	int counter = 0;
	for (int row_index = 0; row_index < depthmap_tar_data.rows; row_index++)
	{
		for (int col_index = 0; col_index < depthmap_tar_data.cols; col_index++)
		{
			depthmap_tar_data.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}
	cv::Mat depthmap_ref_data = depthmap_tar_data * depth_scale + depth_offset;

	depthmap_list.push_back(depthmap_ref_data);
	depthmap_list.push_back(depthmap_tar_data);
	return depthmap_list;
}


void DataImitator::output_date()
{
	// 0) output coefficients to *.json file.
	AlignCoeff aligncoeff;
	aligncoeff.initial(coeff_grid_width, coeff_grid_height, 2);
	aligncoeff.coeff_scale_mat.clear();
	aligncoeff.coeff_scale_mat.push_back(coeff_list[0]);
	aligncoeff.coeff_scale_mat.push_back(coeff_list[2]);

	aligncoeff.coeff_offset_mat.clear();
	aligncoeff.coeff_offset_mat.push_back(coeff_list[1]);
	aligncoeff.coeff_offset_mat.push_back(coeff_list[3]);

	std::vector<std::string> filename_list;
	filename_list.push_back(depthmap_ref_filepath);
	filename_list.push_back(depthmap_tar_filepath);
	aligncoeff.save(json_filepath, filename_list);

	// 1) output to .pfm
	if (!depthmap_ref_filepath.empty() && !depthmap_tar_filepath.empty())
	{
		DepthmapIO::save(depthmap_ref_filepath, depthmap_list[0]);
		DepthmapIO::save(depthmap_tar_filepath, depthmap_list[1]);
	}

	// 2) output pixel corresponding to JSON file.
	if (!depthmap_ref_filepath.empty() && !depthmap_tar_filepath.empty() && !corr_ref2tar_filepath.empty() && !corr_tar2ref_filepath.empty())
	{
		std::string depthmap_ref_filename = fs::path(depthmap_ref_filepath).filename().string();
		std::string depthmap_tar_filename = fs::path(depthmap_tar_filepath).filename().string();
		PixelsCorrIO::save(corr_ref2tar_filepath, depthmap_ref_filename, depthmap_tar_filename, corresponding_mat_ref2tar);
		PixelsCorrIO::save(corr_tar2ref_filepath, depthmap_tar_filename, depthmap_ref_filename, corresponding_mat_tar2ref);
	}
}

std::vector<cv::Mat> DataImitator::make_depthmap_pair_random()
{
	// 0) get random coefficient
	std::vector<cv::Mat> depthmap_pair = make_depthmap_pair_simple();

	// 1) deform the depth map 
	DepthmapUtil::deform(depthmap_pair[0], depthmap_pair[1], coeff_list[2], coeff_list[3]);

	depthmap_list.clear();
	depthmap_list.push_back(depthmap_pair[1]); // reference depth map
	depthmap_list.push_back(depthmap_pair[0]);
	return depthmap_list;
}

std::vector<cv::Mat> DataImitator::make_aligncoeffs_simple()
{
	int ref_image_idx = 0;
	//
	//std::vector<cv::Mat> coeffs_list;
	cv::Mat ref_scale = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1);
	coeff_list.push_back(ref_scale);
	cv::Mat ref_offset = cv::Mat::zeros(coeff_grid_height, coeff_grid_width, CV_64FC1);
	coeff_list.push_back(ref_offset);
	//
	cv::Mat dest_scale = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1) * depth_scale;
	coeff_list.push_back(dest_scale);
	cv::Mat dest_offset = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1) * depth_offset;
	coeff_list.push_back(dest_offset);

	return coeff_list;
}


std::vector<cv::Mat> DataImitator::make_aligncoeffs_random()
{
	coeff_list.clear();
	cv::Mat ref_scale = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1);
	coeff_list.push_back(ref_scale);
	cv::Mat ref_offset = cv::Mat::zeros(coeff_grid_height, coeff_grid_width, CV_64FC1);
	coeff_list.push_back(ref_offset);

	// the random target depth map coefficients
	double scale_rand_max = 4;
	double scale_rand_min = -4;
	cv::Mat dest_scale(coeff_grid_height, coeff_grid_width, CV_64FC1);
	cv::randu(dest_scale, cv::Scalar(scale_rand_min), cv::Scalar(scale_rand_max));
	coeff_list.push_back(dest_scale);

	double offset_rand_max = 10;
	double offset_rand_min = -10;
	cv::Mat dest_offset(coeff_grid_height, coeff_grid_width, CV_64FC1);
	cv::randu(dest_offset, cv::Scalar(offset_rand_min), cv::Scalar(offset_rand_max));
	coeff_list.push_back(dest_offset);
	return coeff_list;
}

std::map<int, std::map<int, cv::Mat>> DataImitator::make_corresponding_json()
{
	// 0) make corresponding relationship mat
	int hight_boundary = depthmap_hight * (1.0 - depthmap_overlap_ratio);
	corresponding_mat_ref2tar = cv::Mat((depthmap_hight - hight_boundary) * depthmap_width, 4, CV_64FC1);
	corresponding_mat_tar2ref = cv::Mat((depthmap_hight - hight_boundary) * depthmap_width, 4, CV_64FC1);

	for (int index_row = 0; index_row < depthmap_hight - hight_boundary; index_row++)
	{
		for (int index_col = 0; index_col < depthmap_width; index_col++)
		{
			int index = index_row * depthmap_width + index_col;
			corresponding_mat_ref2tar.at<double>(index, 0) = hight_boundary + index_row;
			corresponding_mat_ref2tar.at<double>(index, 1) = index_col;
			corresponding_mat_ref2tar.at<double>(index, 2) = index_row;
			corresponding_mat_ref2tar.at<double>(index, 3) = index_col;

			corresponding_mat_tar2ref.at<double>(index, 0) = index_row;
			corresponding_mat_tar2ref.at<double>(index, 1) = index_col;
			corresponding_mat_tar2ref.at<double>(index, 2) = hight_boundary + index_row;
			corresponding_mat_tar2ref.at<double>(index, 3) = index_col;
		}
	}

	// return
	std::map<int, std::map<int, cv::Mat >> pixel_corr;
	std::map<int, cv::Mat> ref2tar;
	ref2tar[1] = corresponding_mat_ref2tar;
	pixel_corr[0] = ref2tar;
	std::map<int, cv::Mat> tar2ref;
	tar2ref[0] = corresponding_mat_tar2ref;
	pixel_corr[1] = tar2ref;
	return pixel_corr;
}

std::ostream& operator<<(std::ostream& os, const DataImitator& di) {
	return os
		<< "Source depth map : " << di.depthmap_ref_filepath << std::endl
		<< "Target depth map : " << di.depthmap_tar_filepath << std::endl
		<< "Source to target depth map : " << di.corr_ref2tar_filepath << std::endl
		<< "Target to source depth map : " << di.corr_tar2ref_filepath << std::endl
		<< "Source depth scale and offset : " << std::to_string(di.depth_scale) << "\t" << std::to_string(di.depth_offset) << std::endl
		<< "Depth map hight and width: " << std::to_string(di.depthmap_hight) << "\t" << std::to_string(di.depthmap_width) << std::endl;
}



void DataImitatorMultiImage::make_aligncoeffs_simple()
{
	//int ref_image_idx = 0;
	////
	////std::vector<cv::Mat> coeffs_list;
	//cv::Mat ref_scale = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1);
	//coeff_list.push_back(ref_scale);
	//cv::Mat ref_offset = cv::Mat::zeros(coeff_grid_height, coeff_grid_width, CV_64FC1);
	//coeff_list.push_back(ref_offset);
	////
	//cv::Mat dest_scale = cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1) * depth_scale;
	//coeff_list.push_back(dest_scale);
	//cv::Mat dest_offset = cv::Mat::zeros(coeff_grid_height, coeff_grid_width, CV_64FC1) * depth_offset;
	//coeff_list.push_back(dest_offset);

	//return coeff_list;
}

void DataImitatorMultiImage::make_aligncoeffs_random()
{
	if (coeff_grid_width * depthmap_overlap_ratio != (int)coeff_grid_width * depthmap_overlap_ratio)
		LOG(ERROR) << "The depth map width should be even.";

	int template_grid_width = coeff_grid_width + coeff_grid_width * depthmap_overlap_ratio * (frame_number - 1);

	// create image [1, frame_number -1]'s image scale & offset coefficients template
	for (int i = 0; i < frame_number - 1; i++)
	{
		cv::Mat scale_template = cv::Mat::ones(coeff_grid_height, template_grid_width, CV_64FC1);
		cv::randn(scale_template, scale_mean, scale_stddev);
		std::map<int, cv::Mat> scale_temp;
		scale_temp[frame_number - 2 - i] = scale_template;
		coeff_template_scale[frame_number - 1 - i] = scale_temp;

		// create offset coefficients template
		cv::Mat offset_template = cv::Mat::ones(coeff_grid_height, template_grid_width, CV_64FC1);
		cv::randn(offset_template, offset_mean, offset_stddev);
		std::map<int, cv::Mat> offset_temp;
		offset_temp[frame_number - 2 - i] = offset_template;
		coeff_template_offset[frame_number - 1 - i] = offset_temp;
	}

	// get image [1, frame_number -1]'s coefficients
	for (int mat_idx = 0; mat_idx < 2; mat_idx++)
	{
		std::map<int, std::map<int, cv::Mat>>& coeff_list = mat_idx == 0 ? coeff_scale_list : coeff_offset_list;

		// coefficients for each image
		for (int i = 0; i < frame_number - 1; i++)
		{
			cv::Mat template_mat = coeff_template_scale.at(frame_number - 1 - i).at(frame_number - 2 - i);
			if (mat_idx == 1)
			{
				template_mat = coeff_template_offset.at(frame_number - 1 - i).at(frame_number - 2 - i);
			}

			// get the each image's coefficients
			cv::Mat coeff_scale_ = cv::Mat(coeff_grid_height, coeff_grid_width, CV_64FC1);
			int col_offset = 0 + coeff_grid_width * depthmap_overlap_ratio * (frame_number - 1 - i);
			for (int row_idx = 0; row_idx < coeff_grid_height; row_idx++)
			{
				for (int col_idx = 0; col_idx < coeff_grid_width; col_idx++)
				{
					int col_idx_template = col_offset + col_idx;
					coeff_scale_.at<double>(row_idx, col_idx) = template_mat.at<double>(row_idx, col_idx_template);
				}
			}

			std::map<int, cv::Mat> tar_mat;
			tar_mat[frame_number - 2 - i] = coeff_scale_;
			coeff_list.insert({ frame_number - 1 - i , tar_mat });
		}
	}

	// the last frame
	coeff_scale_list.at(frame_number - 1).insert({ frame_number - 1, cv::Mat::ones(coeff_grid_height, coeff_grid_width, CV_64FC1) });
	coeff_offset_list.at(frame_number - 1).insert({ frame_number - 1, cv::Mat::zeros(coeff_grid_height, coeff_grid_width, CV_64FC1) });
}


void  DataImitatorMultiImage::make_depthmap_pair()
{
	if (depthmap_width * depthmap_overlap_ratio != (int)depthmap_width * depthmap_overlap_ratio)
		LOG(ERROR) << "The depth map width should be even.";

	// 1) create reference depth map template
	int template_depthmap_width = depthmap_width + depthmap_width * depthmap_overlap_ratio * (frame_number - 1);
	// the template of reference depth map
	cv::Mat depthmap_ref_template = cv::Mat::ones(depthmap_hight, template_depthmap_width, CV_64FC1);
	int counter = 0;
	for (int row_index = 0; row_index < depthmap_ref_template.rows; row_index++)
	{
		for (int col_index = 0; col_index < depthmap_ref_template.cols; col_index++)
		{
			depthmap_ref_template.at<double>(row_index, col_index) = counter;
			counter++;
		}
	}

	// 2) deform the reference depth map to get all template depth map
	depthmap_template_list[frame_number - 1] = depthmap_ref_template;
	for (int frame_idx = 0; frame_idx < frame_number - 1; frame_idx++)
	{
		int ref_mat_idx = frame_number - 1 - frame_idx;
		int tar_mat_idx = frame_number - 2 - frame_idx;
		cv::Mat& ref_mat = depthmap_template_list[ref_mat_idx];
		cv::Mat ref_mat_deformed;
		DepthmapUtil::deform(ref_mat, ref_mat_deformed, coeff_template_scale[ref_mat_idx][tar_mat_idx], coeff_template_offset[ref_mat_idx][tar_mat_idx]);
		depthmap_template_list[tar_mat_idx] = ref_mat_deformed;
	}

	// 2) get sub-depth map
	for (int frame_idx = 0; frame_idx < frame_number; frame_idx++)
	{
		int col_offset = depthmap_width * depthmap_overlap_ratio * frame_idx;
		int col_start = 0 + col_offset;
		int col_end = depthmap_width - 1 + col_offset;

		cv::Mat subdepthmap = cv::Mat::ones(depthmap_hight, depthmap_width, CV_64FC1);
		depthmap_list[frame_idx] = subdepthmap;

		cv::Mat& depthmap_ref_template = depthmap_template_list[frame_idx];
		for (int row_index = 0; row_index < depthmap_hight; row_index++)
		{
			for (int col_index = 0; col_index < depthmap_width; col_index++)
			{
				subdepthmap.at<double>(row_index, col_index) = depthmap_ref_template.at<double>(row_index, col_index + col_offset);
			}
		}
	}
}


void DataImitatorMultiImage::make_pixel_corresponding()
{
	for (int frame_src = 0; frame_src < frame_number; frame_src++)
	{// the source frame index
		for (int frame_tar = 0; frame_tar < frame_number; frame_tar++)
		{ // the target frame index

			// 1) the overlap area column index range.
			int src_col_start = -1;
			int src_col_end = -1;
			int tar_col_start = -1;
			int tar_col_end = -1;

			int src_col_start_template = -1;
			int src_col_end_template = -1;
			int tar_col_start_template = -1;
			int tar_col_end_template = -1;

			// range in src or tar image
			if (frame_src == frame_tar)
				continue;
			else if (frame_src < frame_tar)
			{
				src_col_start = (1.0 - depthmap_overlap_ratio) * depthmap_width;
				src_col_end = depthmap_width;
				tar_col_start = 0;
				tar_col_end = depthmap_overlap_ratio * depthmap_width;
			}
			else if (frame_src > frame_tar)
			{
				src_col_start = 0;
				src_col_end = depthmap_overlap_ratio * depthmap_width;
				tar_col_start = (1.0 - depthmap_overlap_ratio) * depthmap_width;
				tar_col_end = depthmap_width;
			}

			// range in template image
			src_col_start_template = src_col_start + frame_src * (1.0 - depthmap_overlap_ratio) * depthmap_width;
			src_col_end_template = src_col_end + frame_src * (1.0 - depthmap_overlap_ratio) * depthmap_width;
			tar_col_start_template = tar_col_start + frame_tar * (1.0 - depthmap_overlap_ratio) * depthmap_width;
			tar_col_end_template = tar_col_end + frame_tar * (1.0 - depthmap_overlap_ratio) * depthmap_width;

			// check src and tar overlap range
			if (src_col_start_template >= tar_col_end_template || src_col_end_template <= tar_col_start_template)
				continue;

			// 2) create the pixel corresponding relationship
			cv::Mat corre_src2tar = cv::Mat(depthmap_hight * depthmap_width * depthmap_overlap_ratio, 4, CV_64FC1);

			// figure out the corresponding relationship pairs start number
			for (int index_row = 0; index_row < depthmap_hight; index_row++)
			{
				for (int index_col = 0; index_col < depthmap_width * depthmap_overlap_ratio; index_col++)
				{
					int index = index_row * depthmap_width * depthmap_overlap_ratio + index_col;
					corre_src2tar.at<double>(index, 0) = index_row;
					corre_src2tar.at<double>(index, 1) = src_col_start + index_col;
					corre_src2tar.at<double>(index, 2) = index_row;
					corre_src2tar.at<double>(index, 3) = tar_col_start + index_col;
				}
			}

			// 3) store in map
			pixel_corresponding[frame_src].insert({ frame_tar , corre_src2tar });
		}
	}
}

void DataImitatorMultiImage::report_data_parameters()
{
	std::stringstream ss;
	ss << "Depth map frame number: " << frame_number << std::endl;
	ss << "Depth map overlap area ratio:" << depthmap_overlap_ratio << std::endl;
	ss << "Depth map hight and width: " << std::to_string(depthmap_hight) << "\t" << std::to_string(depthmap_width) << std::endl;
	ss << "Coefficients grid size (Height * Width): " << std::to_string(coeff_grid_height) << "\t" << std::to_string(coeff_grid_width) << std::endl;
	LOG(INFO) << ss.str();
}


void DataImitatorMultiImage::output_date()
{
	const std::string filename_prefix;
	const int index_digit_number = 3;

	// 0) output depth map to *.pfm
	std::vector<std::string> pfm_filename_list;
	for (int frame_index = 0; frame_index < frame_number; frame_index++)
	{
		char frame_index_str[32];
		sprintf(frame_index_str, "%03d", frame_index);
		// depth map file name
		std::string depthmap_filepath = std::regex_replace(depthmap_filename_exp, std::regex(R"(\$subimageindex)"), frame_index_str);
		pfm_filename_list.push_back(output_root_dir + depthmap_filepath);
		DepthmapIO::save(output_root_dir + depthmap_filepath, depthmap_list[frame_index]);
	}

	// 1) output coefficients to *.json file.
	// coefficients file path
	AlignCoeff aligncoeff;
	aligncoeff.initial(coeff_grid_width, coeff_grid_height, frame_number);
	aligncoeff.coeff_scale_mat.clear();
	aligncoeff.coeff_offset_mat.clear();
	for (int frame_idx = 0; frame_idx < frame_number - 1; frame_idx++) {
		aligncoeff.coeff_scale_mat.push_back(coeff_scale_list[frame_idx + 1][frame_idx]);
		aligncoeff.coeff_offset_mat.push_back(coeff_offset_list[frame_idx + 1][frame_idx]);
	}
	aligncoeff.coeff_scale_mat.push_back(coeff_scale_list[frame_number - 1][frame_number - 1]);
	aligncoeff.coeff_offset_mat.push_back(coeff_offset_list[frame_number - 1][frame_number - 1]);

	std::string json_filepath = output_root_dir + aligncoeff_filename_exp;
	aligncoeff.save(json_filepath, pfm_filename_list);

	// 2) output pixel corresponding to JSON file.
	// generate the corresponding relation file name
	auto zeropad = [](const int number, const int digit_number)->std::string
	{
		std::ostringstream ss;
		ss << std::setw(digit_number) << std::setfill('0') << number;
		return ss.str();
	};

	for (int frame_src = 0; frame_src < frame_number; frame_src++)
	{// the source frame index
		for (int frame_tar = 0; frame_tar < frame_number; frame_tar++)
		{ // the target frame index
			if (frame_src == frame_tar)
				continue;
			std::string pixels_corr_filepath;
			pixels_corr_filepath = std::regex_replace(pixelcorr_filename_exp, std::regex(R"(\$prefix)"), filename_prefix);
			pixels_corr_filepath = std::regex_replace(pixels_corr_filepath, std::regex(R"(\$srcindex)"), zeropad(frame_src, index_digit_number));
			pixels_corr_filepath = std::regex_replace(pixels_corr_filepath, std::regex(R"(\$tarindex)"), zeropad(frame_tar, index_digit_number));

			cv::Mat corresponding_mat = pixel_corresponding[frame_src][frame_tar];
			std::string depthmap_src_filename = pfm_filename_list[frame_src];
			std::string depthmap_tar_filename = pfm_filename_list[frame_tar];

			PixelsCorrIO::save(output_root_dir + pixels_corr_filepath, depthmap_src_filename, depthmap_tar_filename, corresponding_mat);
		}
	}
}
