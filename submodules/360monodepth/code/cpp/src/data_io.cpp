#include "data_io.hpp"
#include "pfm_io.hpp"

#include <boost/property_tree/json_parser.hpp>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include <iostream>
#include <utility>
#include <string>
#include <regex>

namespace pt = boost::property_tree;

void DataIO::write2json(const std::string& filename, const pt::ptree& root)
{
	std::stringstream ss;
	pt::write_json(ss, root);
	std::string my_string_to_send_somewhere_else = ss.str();

	// remove digital number quotes
	std::regex reg0("\\\"([+-]*[0-9]+\\.{0,1}[0-9]*)\\\"");
	std::string result = std::regex_replace(my_string_to_send_somewhere_else, reg0, "$1");
	std::regex reg1("\\\"([+-]*[0-9]+\\.{0,1}[0-9]*\\e[+-]{0,1}[0-9]*)\\\"");
	result = std::regex_replace(result, reg1, "$1");

	std::ofstream file;
	file.open(filename);
	file << result;
	file.close();
}

void DepthmapIO::load(const std::string& filepath, cv::Mat& depthmap_mat)
{
	PFM pfm_depth;
	float* depthmap_data = pfm_depth.read_pfm<float>(filepath);
	int depth_map_height = pfm_depth.getHeight();
	int depth_map_width = pfm_depth.getWidth();
	depthmap_mat = cv::Mat(depth_map_height, depth_map_width, CV_32FC1);
	memcpy(depthmap_mat.data, depthmap_data, depth_map_height * depth_map_width * sizeof(float));
	depthmap_mat.convertTo(depthmap_mat, CV_64FC1);
	delete depthmap_data;

	//cv::imshow("Depth Map window", depthmap_mat);
	//int k = cv::waitKey(0); // Wait for a keystroke in the window
}

void DepthmapIO::save(const std::string& filepath, const cv::Mat& depthmap_mat)
{
	PFM pfm_depth;
	pfm_depth.setHeight(depthmap_mat.rows);
	pfm_depth.setWidth(depthmap_mat.cols);
	cv::Mat depthmap_mat_float;
	depthmap_mat.convertTo(depthmap_mat_float, CV_32FC1);
	pfm_depth.write_pfm<float>(filepath, (float*)depthmap_mat_float.data, -1.0);
}

void PixelsCorrIO::save(const std::string& file_path, const std::string& src_filename, const std::string& tar_filename, const cv::Mat& pixles_corresponding)
{
	//pt::ptree root;
	pt::basic_ptree<std::string, std::string> root;

	// output mat
	root.put("src_image_filename", src_filename);
	root.put("src_image_sha256", "   ");

	root.put("tar_image_filename", tar_filename);
	root.put("tar_image_sha256", "   ");

	// output mat row number
	root.put<int>("pixel_corresponding_number", pixles_corresponding.rows);
	//root.push_back(std::make_pair("pixel_corresponding_number", pixel_corresponding.rows));

	if (!pixles_corresponding.empty() && pixles_corresponding.cols != 4)
		LOG(ERROR) << "The pixels corresponding column number should be 4!";

	// output data
	pt::ptree matrix_node;
	for (int i = 0; i < pixles_corresponding.rows; i++)
	{
		pt::ptree row;
		for (int j = 0; j < pixles_corresponding.cols; j++)
		{
			pt::basic_ptree<std::string, std::string> cell;
			cell.put_value(pixles_corresponding.at<double>(i, j));
			row.push_back(std::make_pair("", cell));
		}
		matrix_node.push_back(std::make_pair("", row));
	}
	root.add_child("pixel_corresponding", matrix_node);

	// output to file
	DataIO::write2json(file_path, root);
}


void PixelsCorrIO::load(const std::string& file_path, std::string& src_filename, std::string& tar_filename, cv::Mat& pixles_corresponding)
{
	pt::ptree root;
	pt::read_json(file_path, root);

	int pixel_corresponding_number = -1;

	for (pt::ptree::value_type& key_value_pair : root)
	{
		std::string key = key_value_pair.first;
		if (key.compare("src_image_filename") == 0)
		{
			src_filename = key_value_pair.second.data();
		}
		else if (key.compare("tar_image_filename") == 0)
		{
			tar_filename = key_value_pair.second.data();
		}
		else if (key.compare("pixel_corresponding") == 0)
		{
			if (pixel_corresponding_number != -1)
			{
				pixles_corresponding = cv::Mat::zeros(pixel_corresponding_number, 4, CV_64FC1);
				int x = 0;
				for (pt::ptree::value_type& row : root.get_child(key))
				{
					int y = 0;
					for (pt::ptree::value_type& cell : row.second)
					{
						pixles_corresponding.at<double>(x, y) = cell.second.get_value<float>();
						y++;
					}
					x++;
				}
			}
		}
		else if (key.compare("pixel_corresponding_number") == 0)
		{
			pixel_corresponding_number = key_value_pair.second.get_value<int>();
			//printf("%d \n", pixel_corresponding_number);
		}
		else
		{
			LOG(INFO) << "Key  \n" << key;
		}
	}
}
