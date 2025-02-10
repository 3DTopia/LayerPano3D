#pragma once

#include <string>
#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace cv
{
	class Mat;
}


class DataIO
{
public:
	/**
	 * Format the json string and output to file.
	 */
	static void write2json(const std::string& filename, const boost::property_tree::ptree& root);
};

// load and save depth map
class DepthmapIO : public DataIO
{
public:
	static void save(const std::string& filepath, const cv::Mat& depth_mat);

	static void load(const std::string& filepath, cv::Mat& depth_mat);
};

// load and save pixel corresponding ship
class PixelsCorrIO : public DataIO
{
public:
	static void save(const std::string& file_path, const std::string& src_filename, const std::string& tar_filename, const cv::Mat& pixles_corresponding);

	static void load(const std::string& file_path, std::string& src_filename, std::string& tar_filename, cv::Mat& pixles_corresponding);
};
