#include "depthmap_stitcher.hpp"
#include "depthmap_utility.hpp"
#include "data_io.hpp"

#include <glog/logging.h>
#include <boost/property_tree/json_parser.hpp>

#include <iostream>
#include <string>
#include <regex>

namespace pt = boost::property_tree;
using namespace std;

DepthmapStitcher::DepthmapStitcher()
{
}

DepthmapStitcher::~DepthmapStitcher()
{
}

std::string zeropad(const int number, const int digit_number)
{
	std::ostringstream ss;
	ss << std::setw(digit_number) << std::setfill('0') << number;
	return ss.str();
}

void DepthmapStitcher::initial_data(const std::string& data_root_dir,
	const std::vector<cv::Mat>& depthmap_original_data,
	const std::vector<int>& depthmap_original_ico_index,
	const std::map<int, std::map<int, cv::Mat>>& pixels_corresponding_list_data)
{
	if (depthmap_original_data.size() < 2)
		LOG(ERROR) << "The depth map image less than 2.";

	// set the output filename's prefix
	filename_prefix = "depthmapAlignPy";

	this->data_root_dir = data_root_dir;
	// 0) generate the depth maps proxy filename for output debug and report.
	filename_list.clear();
	depthmap_original.clear();
	for (int index_inter = 0; index_inter < depthmap_original_data.size(); index_inter++)
	{
		// create the proxy filename
		int index_exter = depthmap_original_ico_index[index_inter];
		std::string filename = std::regex_replace(proxy_filename_depthmap_exp, std::regex(R"(\$index)"), std::to_string(index_exter));
		LOG(INFO) << "Generate depth map proxy filename:" << filename;
		filename_list.push_back(filename);
		if (depthmap_original_data[index_inter].depth() == CV_64F)
		{
			depthmap_original.push_back(depthmap_original_data[index_inter]);
		}
		else {
			LOG(INFO) << "The aligned depth map is not double cv::Mat, convert to float cv::Mat.";
			cv::Mat depthmap_double;
			depthmap_original_data[index_inter].convertTo(depthmap_double, CV_64F);
			depthmap_original.push_back(depthmap_double);
		}
		extidx2intidx[index_exter] = index_inter;
		intidx2extidx[index_inter] = index_exter;
	}

	// 1) the pixel corresponding relationship, and convert the sub-image index to internal index.
	pixels_corresponding_list.clear();
	for (auto src_iter = pixels_corresponding_list_data.begin(); src_iter != pixels_corresponding_list_data.end(); src_iter++)
	{
		std::map<int, int>::iterator src_intidx_it = extidx2intidx.find(src_iter->first);
		if (src_intidx_it == extidx2intidx.end())
		{
			LOG(WARNING) << "src_intidx " << src_iter->first << " do not exist!";
			continue;
		}
		int src_intidx = src_intidx_it->second;

		std::map<int, cv::Mat> tar_map = src_iter->second;
		for (auto tar_map_iter = tar_map.begin(); tar_map_iter != tar_map.end(); tar_map_iter++)
		{
			std::map<int, int>::iterator tar_intidx_it = extidx2intidx.find(tar_map_iter->first);
			if (tar_intidx_it == extidx2intidx.end())
			{
				LOG(WARNING) << "tar_intidx " << tar_map_iter->first << " do not exist!";
				continue;
			}
			int tar_intidx = tar_intidx_it->second;
			tar_map[tar_intidx] = tar_map_iter->second;
		}
		pixels_corresponding_list[src_intidx] = tar_map;
	}
	// check the corresponding relationship data
}

void DepthmapStitcher::load_data(const std::string& data_root_dir, const std::vector<std::string> depthmap_filename_list)
{
	if (depthmap_filename_list.size() < 2)
		LOG(ERROR) << "The depth map image less than 2.";

	this->data_root_dir = data_root_dir;
	// 0) test and parser the file name, get the mapping between external index and internal index
	filename_list.clear();
	std::map<int, std::string> extidx2filename;
	// for (std::string &filename : depthmap_filename_list)
	for (int index = 0; index < depthmap_filename_list.size(); index++)
	{
		std::string filename = depthmap_filename_list[index];
		filename_list.push_back(filename);

		if (!std::regex_match(filename, std::regex(depthmap_filename_regexp)))
			LOG(ERROR) << "Input file name " << filename << " do not match the file name expression" << depthmap_filename_regexp << " please check.";

		// parser the file name get sub-images external index
		std::smatch m;
		regex_search(filename, m, std::regex("_[0-9]+.pfm"));
		std::string ext_index_str = m[0].str();
		ext_index_str = ext_index_str.substr(1, ext_index_str.length() - 4);
		int depthmap_extindex = std::stoi(ext_index_str);

		extidx2filename[depthmap_extindex] = filename;
		extidx2intidx[depthmap_extindex] = index;
		intidx2extidx[index] = depthmap_extindex;
	}
	smatch m_prefixs;
	regex_search(depthmap_filename_list[0], m_prefixs, std::regex("[a-zA-Z0-9\\_]*\\_disp"));
	std::string prefix = m_prefixs[0].str();
	filename_prefix = prefix.substr(0, prefix.length()-5);
	LOG(INFO) << "File name prefix is " << filename_prefix;

	// 1) load the pixel corresponding relationship
	pixels_corresponding_list.clear();
	for (std::map<int, std::string>::iterator it_src = extidx2filename.begin(); it_src != extidx2filename.end(); ++it_src)
	{
		//disparitmap_index_list.push_back(index_src);
		int extidx_src = it_src->first;
		int intidx_src = extidx2intidx[extidx_src];
		std::map<int, cv::Mat> pixels_corresponding_sublist;

		for (std::map<int, std::string>::iterator it_tar = extidx2filename.begin(); it_tar != extidx2filename.end(); ++it_tar)
		{
			int extidx_tar = it_tar->first;
			if (extidx_src == extidx_tar)
				continue;

			int intidx_tar = extidx2intidx[extidx_tar];
			// generate the corresponding relation file name
			std::string pixels_corr_filepath;
			pixels_corr_filepath = std::regex_replace(pixelcorr_filename_exp, std::regex(R"(\$prefix)"), filename_prefix);
			pixels_corr_filepath = std::regex_replace(pixels_corr_filepath, std::regex(R"(\$srcindex)"), zeropad(extidx_src, index_digit_number));
			pixels_corr_filepath = std::regex_replace(pixels_corr_filepath, std::regex(R"(\$tarindex)"), zeropad(extidx_tar, index_digit_number));

			// load pixel corresponding relationship
			LOG(INFO) << "Loading corresponding file :" << pixels_corr_filepath;
			std::string src_filename;
			std::string tar_filename;
			cv::Mat pixles_corresponding;
			PixelsCorrIO::load(data_root_dir + pixels_corr_filepath, src_filename, tar_filename, pixles_corresponding);
			pixels_corresponding_sublist[intidx_tar] = pixles_corresponding;

			// check the filename
			if (src_filename.compare(filename_list[intidx_src]) != 0)
				LOG(ERROR) << "The pixels corresponding file " << pixels_corr_filepath << " reference image filename " << src_filename << "is not same as expect " << filename_list[intidx_src];
			if (tar_filename.compare(filename_list[intidx_tar]) != 0)
				LOG(ERROR) << "The pixels corresponding file " << pixels_corr_filepath << " target image filename " << tar_filename << "is not same as expect " << filename_list[intidx_tar];
		}
		pixels_corresponding_list[intidx_src] = pixels_corresponding_sublist;
	}

	// 2) load depth map base on the corresponding relationship
	depthmap_original.clear();
	for (const std::string& filename : depthmap_filename_list)
	{
		std::string depth_map_filepath = data_root_dir + filename;
		cv::Mat depthmap;
		DepthmapIO::load(depth_map_filepath, depthmap);
		if (depthmap.size().area() == 0)
			LOG(ERROR) << "The depth map " << depth_map_filepath << " is empty.";
		depthmap_original.push_back(depthmap);
		LOG(INFO) << "Load depth map from :" << depth_map_filepath;
	}
}

void DepthmapStitcher::initial(const int grid_width, const int grid_height)
{
	if (depthmap_original.size() != filename_list.size())
		LOG(ERROR) << "The depth loading is not completed, please load the data again!";

	// 1) initial the optimized parameters
	this->grid_width = grid_width;
	this->grid_height = grid_height;
	coeff_so.initial(grid_width, grid_height, depthmap_original.size());
}

void DepthmapStitcher::set_coeff(const std::vector<cv::Mat>& coeff_scale,
	const std::vector<cv::Mat>& coeff_offset)
{
	coeff_so.set_value_mat(coeff_scale, coeff_offset);
}

void DepthmapStitcher::align_depthmap_all()
{
	depthmap_aligned.resize(depthmap_original.size());

	// align each depth maps
	for (int index = 0; index < depthmap_original.size(); index++)
	{
		align_depthmap(index);
	}
}

void DepthmapStitcher::align_depthmap(const int depthmap_intidx)
{
	if (depthmap_aligned.size() != depthmap_original.size())
		depthmap_aligned.resize(depthmap_original.size());

	DepthmapUtil::deform(depthmap_original[depthmap_intidx], depthmap_aligned[depthmap_intidx],
		coeff_so.coeff_scale_mat[depthmap_intidx], coeff_so.coeff_offset_mat[depthmap_intidx]);
}

void DepthmapStitcher::get_align_coeff(std::vector<cv::Mat>& coeff)
{
	for (int i = 0; i < coeff_so.coeff_scale_mat.size(); i++)
	{
		
		coeff.push_back(coeff_so.coeff_scale_mat[i].clone());
		coeff.push_back(coeff_so.coeff_offset_mat[i].clone());
	}
}

void DepthmapStitcher::save_align_coeff()
{
	const std::string coeff_json_filepath = data_root_dir + filename_prefix + "_coeff.json";
	LOG(INFO) << "Output the aligned coefficients to file :" << coeff_json_filepath;
	coeff_so.save(coeff_json_filepath, filename_list);
}

void DepthmapStitcher::save_aligned_depthmap()
{
	for (int index = 0; index < depthmap_aligned.size(); index++)
	{
		int depthmap_index = intidx2extidx[index];
		std::string output_pfm_filepath = std::regex_replace(depthmap_aligned_filename_exp, std::regex(R"(\$prefix)"), filename_prefix);
		output_pfm_filepath = std::regex_replace(output_pfm_filepath, std::regex(R"(\$subindex)"), zeropad(depthmap_index, index_digit_number));
		output_pfm_filepath = data_root_dir + output_pfm_filepath;
		LOG(INFO) << "Save aligned depth map :" << output_pfm_filepath;
		DepthmapIO::save(output_pfm_filepath, depthmap_aligned[index]);
	}
}

void DepthmapStitcher::report_error()
{
	if (depthmap_aligned.size() != depthmap_original.size())
		LOG(ERROR) << "The aligned map is not completed.";

	// compute aligned depth maps RMS
	std::stringstream ss;
	// TODO multi-thread with openmp
	for (std::map<int, int>::iterator iter_src = intidx2extidx.begin(); iter_src != intidx2extidx.end(); ++iter_src)
	{
		int intidx_src = iter_src->first;
		int extidx_src = iter_src->second;
		const cv::Mat& depthmap_src = depthmap_aligned[intidx_src];
		std::map<int, cv::Mat> tar_map = pixels_corresponding_list[intidx_src];

		for (std::map<int, int>::iterator iter_tar = intidx2extidx.begin(); iter_tar != intidx2extidx.end(); ++iter_tar)
		{
			int intidx_tar = iter_tar->first;
			int extidx_tar = iter_tar->second;

			if (intidx_tar == intidx_src)
			{
				ss << extidx_src << "->" << extidx_tar << ":" << 0 << "\t";
				continue;
			}

			const cv::Mat& depthmap_tar = depthmap_aligned[intidx_tar];
			const cv::Mat& corr_src2tar = tar_map.at(intidx_tar);

			// compute RMS
			float rms = 0;
			int valuable_pixel_counter = 0;
			for (int i = 0; i < corr_src2tar.rows; i++)
			{
				valuable_pixel_counter++;

				int src_y = corr_src2tar.at<double>(i, 0);
				int src_x = corr_src2tar.at<double>(i, 1);
				int tar_y = corr_src2tar.at<double>(i, 2);
				int tar_x = corr_src2tar.at<double>(i, 3);

				float depth_value_src = getColorSubpix(depthmap_src, cv::Point2f(src_x, src_y));
				float depth_value_tar = getColorSubpix(depthmap_tar, cv::Point2f(tar_x, tar_y));

				rms += (depth_value_src - depth_value_tar) * (depth_value_src - depth_value_tar);
			}
			if (valuable_pixel_counter != 0)
				rms = std::sqrt(rms / valuable_pixel_counter);
			else
				rms = -1;
			//LOG(INFO) << "RMS between depth map \n"
			//	<< filename_list[index_src] << " and " << filename_list[index_tar]
			//	<< "\n is :" << rms;
			ss << extidx_src << "->" << extidx_tar << ":" << rms << "\t";
		}
		ss << std::endl << "-----" << std::endl;
	}
	LOG(INFO) << "RMS Error : \n " << ss.str();
}

float DepthmapStitcher::getColorSubpix(const cv::Mat &img, const cv::Point2f pt)
{
	// cv::Mat patch;
	// assert(img.depth() == CV_64FC1);
	// assert(img.channels() == 1);
	// cv::Mat imgF;
	// img.convertTo(imgF, CV_32FC1);
	// cv::getRectSubPix(imgF, cv::Size(1, 1), pt, patch);
	// return patch.at<float>(0, 0);

	int x = (int)pt.x;
	int y = (int)pt.y;
	int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
	int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
	int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
	int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

	float a = pt.x - (float)x;
	float c = pt.y - (float)y;

	return (double)cvRound(
		(img.at<double>(y0, x0) * (1.f - a) + img.at<double>(y0, x1) * a) * (1.f - c) +
		(img.at<double>(y1, x0) * (1.f - a) + img.at<double>(y1, x1) * a) * c);
}

void DepthmapStitcher::get_bilinear_weight(double* weight_list,
	const int image_width, const int image_height,
	const int grid_row_number, const int grid_col_number,
	const double x, const double y)
{
	DepthmapUtil::bilinear_weight(weight_list,
		image_width, image_height,
		grid_col_number, grid_row_number,
		x, y);
}

void AlignCoeff::initial(const int grid_width, const int grid_height, const int depthmap_number)
{
	coeff_number = depthmap_number;
	coeff_cols = grid_width;
	coeff_rows = grid_height;

	// allocate continued memory
	int mem_size = grid_width * grid_height * depthmap_number;
	coeff_scale = std::shared_ptr<double[]>(new double[mem_size]);
	memset(coeff_scale.get(), 0, sizeof(double) * mem_size);
	coeff_offset = std::shared_ptr<double[]>(new double[mem_size]);
	memset(coeff_offset.get(), 0, sizeof(double) * mem_size);

	for (int index = 0; index < coeff_number; index++)
	{
		cv::Mat s_param_mat = cv::Mat(coeff_rows, coeff_cols, CV_64FC1, coeff_scale.get() + coeff_cols * coeff_rows * index);
		coeff_scale_mat.push_back(s_param_mat);
		cv::Mat o_param_mat = cv::Mat(coeff_rows, coeff_cols, CV_64FC1, coeff_offset.get() + coeff_cols * coeff_rows * index);
		coeff_offset_mat.push_back(o_param_mat);
	}
}


void AlignCoeff::set_value_const(const float scale, const float offset)
{
	for (int index = 0; index < coeff_number; index++)
	{
		coeff_scale_mat[index].setTo(scale);
		coeff_offset_mat[index].setTo(offset);;
	}
}


void AlignCoeff::set_value_mat(const std::vector<cv::Mat>& coeff_scale, const std::vector<cv::Mat>& coeff_offset)
{
	// check the data type and number
	if (coeff_scale.size() != coeff_scale_mat.size() ||
		coeff_offset.size() != coeff_offset_mat.size())
	{
		LOG(ERROR) << "The initial coefficients mat size is not equal the internal size.";
	}

	if (coeff_scale[0].depth() != CV_64F || coeff_offset[0].depth() != CV_64F)
	{
		LOG(ERROR) << "The initial coefficients mat depth should be double (CV_64F).";
	}

	// set the initial value
	for (int index = 0; index < coeff_scale.size(); index++)
	{
		std::memcpy(coeff_scale_mat[index].data, coeff_scale[index].data, coeff_scale[index].total() * sizeof(double));
		std::memcpy(coeff_offset_mat[index].data, coeff_offset[index].data, coeff_offset[index].total() * sizeof(double));
	}
}


void AlignCoeff::set_value_rand()
{
	double mean = 0.0;
	double stddev = 4.0 / 3.0; // 99.7% of values will be inside [-4, +4] interval
	for (int index = 0; index < coeff_number; index++)
	{
		cv::randn(coeff_scale_mat[index], cv::Scalar(mean), cv::Scalar(stddev));
		cv::randn(coeff_offset_mat[index], cv::Scalar(mean), cv::Scalar(stddev));
	}
}


void AlignCoeff::save(const std::string& file_path, const std::vector<std::string>& filename_list)
{
	pt::ptree root;

	// output mat
	root.put("storage_order", "row_major");
	for (int index = 0; index < coeff_number * 2; index++)
	{
		const int depthmap_index = int(index / 2.0);

		pt::ptree cell;
		cv::Mat param_mat;
		if (index % 2 == 0)
		{
			cell.put("coeff_type", "scale");
			param_mat = coeff_scale_mat[depthmap_index];
		}
		else
		{
			cell.put("coeff_type", "offset");
			param_mat = coeff_offset_mat[depthmap_index];
		}

		cell.put("filename", filename_list[depthmap_index]);

		// output mat column number
		cell.put("mat_width", param_mat.cols);

		// output mat row number
		cell.put("mat_hight", param_mat.rows);

		// output data
		pt::ptree matrix_node;
		for (int i = 0; i < param_mat.rows; i++)
		{
			pt::ptree row;
			for (int j = 0; j < param_mat.cols; j++)
			{
				pt::ptree cell;
				cell.put_value(param_mat.at<double>(i, j));
				row.push_back(std::make_pair("", cell));
			}
			matrix_node.push_back(std::make_pair("", row));
		}
		cell.add_child("mat_data", matrix_node);

		// output mat description
		std::string name;
		if (index % 2 == 0)
			name = std::string("scale mat of ") + std::to_string(int(index / 2));
		else
			name = std::string("offset mat of ") + std::to_string(int(index / 2));
		cell.put("description", name);

		std::string coeff_name = "coeff_mat_" + std::to_string(index);
		root.add_child(coeff_name, cell);
	}

	DataIO::write2json(file_path, root);
}


void AlignCoeff::load(const std::string& file_path)
{
	// read from disk
	pt::ptree root;
	pt::read_json(file_path, root);

	// parser *.json 
	int scale_mat_counter = -1;
	int offset_mat_counter =-1;
	for (pt::ptree::value_type& key_value_pair : root)
	{
		int mat_width = -1;
		int mat_height = -1;
		std::string key = key_value_pair.first;

		bool scale_mat = false;
		bool offset_mat = false;

		if (key.compare("storage_order") == 0)
		{
			//
		}
		else if (key.find("coeff_mat_") != std::string::npos)
		{
			// parser the coefficient 
			for (pt::ptree::value_type& mat_term : root.get_child(key))
			{
				std::string key = mat_term.first;
				if (key.compare("coeff_type") == 0)
				{
					std::string mat_data_type = mat_term.second.data();
					if (mat_data_type.compare("scale") == 0)
					{
						scale_mat = true;
						scale_mat_counter++;
					}
					else if (mat_data_type.compare("offset") == 0)
					{
						offset_mat = true;
						offset_mat_counter++;
					}
					else
						LOG(ERROR) << "Can not parser the matrix type:" << mat_data_type;
				}
				else if (key.compare("filename") == 0)
				{
					DLOG(INFO) << "Parser the depth map " << mat_term.second.data() << " deforming coefficients.";
				}
				else if (key.compare("mat_width") == 0)
				{
					mat_width = std::stoi(mat_term.second.data());
				}
				else if (key.compare("mat_hight") == 0)
				{
					mat_height = std::stoi(mat_term.second.data());
				}
				else if (key.compare("mat_data") == 0)
				{

					if (mat_width == -1 || mat_height == -1)
						LOG(ERROR) << "JSON file parser error" + file_path;

					cv::Mat mat_data_json = cv::Mat::zeros(mat_height, mat_width, CV_64FC1);
					int x = 0;
					for (pt::ptree::value_type& mat_row : mat_term.second)
					{
						int y = 0;
						for (pt::ptree::value_type& mat_col : mat_row.second)
						{
							mat_data_json.at<double>(x, y) = mat_col.second.get_value<float>();
							y++;
						}
						x++;
					}

					// copy to internal mat data
					cv::Mat coeff_data;
					if (scale_mat)
						coeff_data = coeff_scale_mat[scale_mat_counter];
					else if (offset_mat)
						coeff_data = coeff_offset_mat[offset_mat_counter];

					int byte_size = mat_width * mat_height * sizeof(double);
					// memcpy_s(coeff_data.data, coeff_data.total() * coeff_data.elemSize() 
					// 	, mat_data_json.data, coeff_data.total() * coeff_data.elemSize());
					memcpy(coeff_data.data, mat_data_json.data, coeff_data.total() * coeff_data.elemSize());
				}
				else
				{
					std::cout << "Key  \n"
						<< key;
				}
			}
		}
	}
}


std::ostream& operator<<(std::ostream& os, const AlignCoeff& di)
{
	for (int index = 0; index < di.coeff_number * 2; index++)
	{
		const int depthmap_index = int(index / 2.0);
		os << "Coefficients matrix index : " << depthmap_index;
		cv::Mat param_mat;
		if (index % 2 == 0)
		{
			os << "\t coeff_type: scale" << std::endl;
			param_mat = di.coeff_scale_mat[depthmap_index];
		}
		else
		{
			os << "\t coeff_type: offset" << std::endl;
			param_mat = di.coeff_offset_mat[depthmap_index];
		}
		// output mat column number
		os << "mat_width" << param_mat.cols << std::endl;
		// output mat row number
		os << "mat_hight" << param_mat.rows << std::endl;
		// output data
		os << "mat_data" << param_mat << std::endl;
	}
	return os;
}

