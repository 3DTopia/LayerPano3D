#include "depthmap_stitcher_group.hpp"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include <vector>
#include <map>
#include <string>

DepthmapStitcherGroup::DepthmapStitcherGroup() {}

DepthmapStitcherGroup::~DepthmapStitcherGroup() {}


//struct DepthmapStitcherGroup::ReprojectionResidual_fixed {
//	ReprojectionResidual_fixed(double* bilinear_weight_list_tar, 
//		double depth_value_tar, double depth_value_ref,
//		int grid_width, int grid_height) :
//		bilinear_weight_list_tar_(bilinear_weight_list_tar),
//		depth_value_tar_(depth_value_tar), depth_value_ref_(depth_value_ref),
//		grid_width_(grid_width), grid_height_(grid_height) {}
//
//	template <typename T>
//	bool operator()(T const* const* scale_offset_list, T* residual) const
//	{
//		const T* scale_list_tar = scale_offset_list[0];
//		const T* offset_list_tar = scale_offset_list[1];
//		T scale_tar(0);
//		T offset_tar(0);
//		for (int index = 0; index < grid_width_ * grid_height_; index++)
//		{
//			scale_tar += scale_list_tar[index] * bilinear_weight_list_tar_[index];
//			offset_tar += offset_list_tar[index] * bilinear_weight_list_tar_[index];
//		}
//		T temp = (depth_value_tar_ * scale_tar + offset_tar) - depth_value_ref_ ;
//		residual[0] = temp * temp;
//		return true;
//	}
//
//private:
//	const double depth_value_ref_; // the depth value of source depth map
//	const double depth_value_tar_; // the depth value of corresponding pixel in reference depth map
//	const double* bilinear_weight_list_tar_; // weight of S and A, length is grid_x * grid_y
//	const int grid_width_; // the weight grid along the x axis
//	const int grid_height_; // the weight grid along the y axis
//};

// TODO capsule more function to refer Ceres demo. 
struct DepthmapStitcherGroup::ReprojectionResidual {
	ReprojectionResidual(double* bilinear_weight_list_tar, double* bilinear_weight_list_src,
		double depth_value_tar, double depth_value_src,
		int grid_width, int grid_height) :
		bilinear_weight_list_tar_(bilinear_weight_list_tar), bilinear_weight_list_src_(bilinear_weight_list_src),
		depth_value_tar_(depth_value_tar), depth_value_src_(depth_value_src),
		grid_width_(grid_width), grid_height_(grid_height) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list_src = scale_offset_list[0];
		const T* offset_list_src = scale_offset_list[1];

		const T* scale_list_tar = scale_offset_list[2];
		const T* offset_list_tar = scale_offset_list[3];

		T scale_src(0);
		T offset_src(0);
		T scale_tar(0);
		T offset_tar(0);

		for (int index = 0; index < grid_width_ * grid_height_; index++)
		{
			scale_src += scale_list_src[index] * bilinear_weight_list_src_[index];
			offset_src += offset_list_src[index] * bilinear_weight_list_src_[index];

			scale_tar += scale_list_tar[index] * bilinear_weight_list_tar_[index];
			offset_tar += offset_list_tar[index] * bilinear_weight_list_tar_[index];
		}
		T temp = (depth_value_tar_ * scale_tar + offset_tar) - (depth_value_src_ * scale_src + offset_src);
		//T temp = (depth_value_tar_ * scale_tar + offset_tar) - depth_value_src_ ;
		residual[0] = temp * temp;
		return true;
	}

private:
	const double depth_value_src_; // the depth value of source depth map
	const double depth_value_tar_; // the depth value of corresponding pixel in reference depth map
	const double* bilinear_weight_list_tar_; // weight of S and A, length is grid_x * grid_y
	const double* bilinear_weight_list_src_; // weight of S and A, length is grid_x * grid_y
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
};

// @see hedman2018instant:equ_6
struct DepthmapStitcherGroup::SmoothnessResidual_S {
	SmoothnessResidual_S(int* index_current, int* index_neighbour, int edge_number, int grid_x, int grid_y) :
		grid_width_(grid_x), grid_height_(grid_y),
		index_current_(index_current), index_neighbour_(index_neighbour), edge_number_(edge_number) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		//const T* offset_list = scale_offset_list[1];
		T sum_value(0);
		for (int index = 0; index < edge_number_; index++)
		{
			int curr_index = index_current_[index];
			int neig_index = index_neighbour_[index];
			T scale_diff = scale_list[curr_index] - scale_list[neig_index];
			//T offset_diff = offset_list[curr_index] - offset_list[neig_index];
			sum_value += scale_diff * scale_diff;// +offset_diff * offset_diff;
		}
		residual[0] = sum_value;
		return true;
	}

private:
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
	const int* index_current_; // the weight grid along the x axis
	const int* index_neighbour_; // the weight grid along the y axis
	const int edge_number_;
};


// @see hedman2018instant:equ_6
struct DepthmapStitcherGroup::SmoothnessResidual_O {
	SmoothnessResidual_O(int* index_current, int* index_neighbour, int edge_number, int grid_x, int grid_y) :
		grid_width_(grid_x), grid_height_(grid_y),
		index_current_(index_current), index_neighbour_(index_neighbour), edge_number_(edge_number) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		const T* offset_list = scale_offset_list[1];
		T sum_value(0);
		for (int index = 0; index < edge_number_; index++)
		{
			int curr_index = index_current_[index];
			int neig_index = index_neighbour_[index];
			//T scale_diff = scale_list[curr_index] - scale_list[neig_index];
			T offset_diff = offset_list[curr_index] - offset_list[neig_index];
			sum_value +=  offset_diff * offset_diff;
		}
		residual[0] = sum_value;
		return true;
	}

private:
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
	const int* index_current_; // the weight grid along the x axis
	const int* index_neighbour_; // the weight grid along the y axis
	const int edge_number_;
};

// @see hedman2018instant:equ_6
struct DepthmapStitcherGroup::SmoothnessResidual {
	SmoothnessResidual(int* index_current, int* index_neighbour, int edge_number, int grid_x, int grid_y) :
		grid_width_(grid_x), grid_height_(grid_y),
		index_current_(index_current), index_neighbour_(index_neighbour), edge_number_(edge_number) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		const T* offset_list = scale_offset_list[1];
		T sum_value(0);
		for (int index = 0; index < edge_number_; index++)
		{
			int curr_index = index_current_[index];
			int neig_index = index_neighbour_[index];
			T scale_diff = scale_list[curr_index] - scale_list[neig_index];
			T offset_diff = offset_list[curr_index] - offset_list[neig_index];
			sum_value += scale_diff * scale_diff + offset_diff * offset_diff;
		}
		residual[0] = sum_value;
		return true;
	}

private:
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
	const int* index_current_; // the weight grid along the x axis
	const int* index_neighbour_; // the weight grid along the y axis
	const int edge_number_;
};

// @see hedman2018instant:equ_7
struct DepthmapStitcherGroup::ScaleResidual {
	ScaleResidual(int grid_width, int grid_height) :
		grid_width_(grid_width), grid_height_(grid_height) {}

	template <typename T>
	bool operator()(T const* const* scale_list_list, T* residual) const
	{
		const T* scale_list = scale_list_list[0];
		T sum_number(0);
		for (int index = 0; index < grid_width_ * grid_height_; index++)
		{
			sum_number += 1.0 / (scale_list[index] + T(1e-10));
		}
		residual[0] = sum_number;
//      if (ceres::isinf(residual[0]) || ceres::isnan(residual[0])){
//          printf("INFINITY SCALE\n");
//      }
		return true;
	}

private:
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
};

void DepthmapStitcherGroup::initial(const int grid_width, const int grid_height)
{
	if (depthmap_original.size() < 2)
		throw std::runtime_error("The depth map image less than 2.");

	DepthmapStitcher::initial(grid_width, grid_height);

	// the first depth as the reference depth map
	coeff_so.set_value_const(1.0, 0.0);
}

void DepthmapStitcherGroup::compute_align_coeff()
{
	// 0) load set the data and parameters
	double* s_ij_list = coeff_so.coeff_scale.get();
	double* o_ij_list = coeff_so.coeff_offset.get();

	int image_width = depthmap_original[0].cols;
	int image_height = depthmap_original[0].rows;

	// set the reference depth map scale and offset 
	int depthmap_ref_intidx = extidx2intidx[depthmap_ref_extidx];
	coeff_so.coeff_scale_mat[depthmap_ref_intidx].setTo(1.0);
	coeff_so.coeff_offset_mat[depthmap_ref_intidx].setTo(0.0);

	// 1) make the energy function
	// 1-1) re-projection term
	LOG(INFO) << "Adding re-projection term" << std::endl;
	ceres::Problem problem;
	std::vector<void* > bilinear_weight_list_mem;
	std::vector<std::pair<int, int>> ignore_image_pair;

	// add the parameter block & lock the parameters of reference block
	for (int i = 0; i < coeff_so.coeff_scale_mat.size(); i++)
	{
		problem.AddParameterBlock((double*)coeff_so.coeff_scale_mat[i].data, grid_height * grid_width);
		problem.AddParameterBlock((double*)coeff_so.coeff_offset_mat[i].data, grid_height * grid_width);
	}
	if (depthmap_ref_extidx > 0)
	{
		LOG(INFO) << "Fix the reference frame " << depthmap_ref_extidx << " deformation coefficients.";
		problem.SetParameterBlockConstant(s_ij_list + depthmap_ref_intidx * grid_width * grid_height);
		problem.SetParameterBlockConstant(o_ij_list + depthmap_ref_intidx * grid_width * grid_height);
	}

	//std::vector<int> constant_translation;
	//for (int idx = 0; idx < grid_width * grid_height; idx++)
	//	constant_translation.push_back(idx);
	//ceres::SubsetParameterization* subset_parameterization = new ceres::SubsetParameterization(grid_width * grid_height, constant_translation);
	//problem.SetParameterization((double*)coeff_so.coeff_scale_mat[depthmap_ref_intidx].data, subset_parameterization);
	//problem.SetParameterization((double*)coeff_so.coeff_offset_mat[depthmap_ref_intidx].data, subset_parameterization);

	if (projection_per_pixelcost_enable)
	{
		unsigned int pixel_corr_num = 0;
		for (int depthmap_index_src = 0; depthmap_index_src < depthmap_original.size(); depthmap_index_src++)
			for (int depthmap_index_tar = 0; depthmap_index_tar < depthmap_original.size(); depthmap_index_tar++)
			{
				if (depthmap_index_tar == depthmap_index_src)
					continue;
				pixel_corr_num += pixels_corresponding_list.at(depthmap_index_src).at(depthmap_index_tar).rows;
			}

		weight_reprojection = weight_reprojection * ( 1.0 / pixel_corr_num);
		LOG(INFO) << "Enable perpixel reprojection weight, pixel_corr_num is " << pixel_corr_num << ", term weight is " << weight_reprojection;
	}
	// TODO allocate memory once and assign pointer for each corresponding weights
	
	if (smooth_pergrid_enable)
	{
		unsigned int smooth_gird_numb = 0;
		smooth_gird_numb = depthmap_original.size() * grid_height * grid_width;
		weight_smooth = weight_smooth * (1.0 / smooth_gird_numb);
		LOG(INFO) << "Enable pergrid smooth weight, grid_numb is " << smooth_gird_numb << ", term weight is " << weight_smooth;
	}

	// adjusted depth map register to reference depth map to compute the scale and offset
	int omp_num_threads = omp_get_max_threads() - 2;
	LOG(INFO) << "Build ceres problem with " << omp_num_threads << " threads.";
	#pragma omp parallel for ordered schedule(dynamic) num_threads(omp_num_threads)
	for (int depthmap_index_src = 0; depthmap_index_src < depthmap_original.size(); depthmap_index_src++)
	{
		DLOG(INFO) << "Adding the " << depthmap_index_src << " depth alignment information to problem."; 
		const cv::Mat& depth_map_src = depthmap_original[depthmap_index_src];
		//LOG(INFO) << "Target depth map" << depthmap_index_tar;
		for (int depthmap_index_tar = 0; depthmap_index_tar < depthmap_original.size(); depthmap_index_tar++)
		{
			if (depthmap_index_tar == depthmap_index_src)
				continue;

			// pixels corresponding relationship
			const cv::Mat& pixels_corresponding = pixels_corresponding_list.at(depthmap_index_src).at(depthmap_index_tar);
			int observation_pairs_number = pixels_corresponding.rows;
			if (observation_pairs_number == 0)
			{
				#pragma omp critical(ignore_image_pair)
				{
					ignore_image_pair.push_back(std::pair<int, int>(depthmap_index_src, depthmap_index_tar));
				}
				continue;
			}

			// adjusted depth map
			const cv::Mat& depth_map_tar = depthmap_original[depthmap_index_tar];
			//LOG(INFO) << "Source depth map:" << depthmap_index_src;

			// grid interpolation weight for each pixel, row-major
			double* bilinear_weight_list_src = (double*)malloc(grid_width * grid_height * observation_pairs_number * sizeof(double));
			// memset(bilinear_weight_list_src, 0, grid_width * grid_height * observation_pairs_number * sizeof(double));
			double* bilinear_weight_list_tar = (double*)malloc(grid_width * grid_height * observation_pairs_number * sizeof(double));
			// memset(bilinear_weight_list_tar, 0, grid_width * grid_height * observation_pairs_number * sizeof(double));

			#pragma omp critical(bilinear_weight_list_mem)
			{
				bilinear_weight_list_mem.push_back((void*)bilinear_weight_list_src);
				bilinear_weight_list_mem.push_back((void*)bilinear_weight_list_tar);
			}

			// add the depth value and weight value
			for (int observations_index = 0; observations_index < observation_pairs_number; observations_index++)
			{
				const double y_src = pixels_corresponding.at<double>(observations_index, 0);
				const double x_src = pixels_corresponding.at<double>(observations_index, 1);
				const double y_tar = pixels_corresponding.at<double>(observations_index, 2);
				const double x_tar = pixels_corresponding.at<double>(observations_index, 3);

				double depth_value_src = getColorSubpix(depth_map_src, cv::Point2f(x_src, y_src));
				double depth_value_tar = getColorSubpix(depth_map_tar, cv::Point2f(x_tar, y_tar));

				// compute the bilinear weights;
				double* bilinear_weight_src = bilinear_weight_list_src + grid_width * grid_height * observations_index;
				get_bilinear_weight(bilinear_weight_src, image_width, image_height, grid_height, grid_width, x_src, y_src);
				double* bilinear_weight_tar = bilinear_weight_list_tar + grid_width * grid_height * observations_index;
				get_bilinear_weight(bilinear_weight_tar, image_width, image_height, grid_height, grid_width, x_tar, y_tar);

				//// for debug visuzlize the cv::Mat
				//cv::Mat bilinear_weight_list_mat_src(cv::Size( coeff_so.coeff_cols, coeff_so.coeff_rows), CV_64FC1, bilinear_weight_src);
				//cv::Mat bilinear_weight_list_mat_tar(cv::Size(coeff_so.coeff_cols, coeff_so.coeff_rows), CV_64FC1, bilinear_weight_tar);

				// add residual block 
				//if (depthmap_index_tar != depthmap_ref_intidx && depthmap_index_src != depthmap_ref_intidx)
				{
					// TODO The cauchyless is made bad result. figure our reason.
					//ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(new ceres::CauchyLoss(1), weight_re-projection, ceres::TAKE_OWNERSHIP);
					ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(nullptr, weight_reprojection, ceres::TAKE_OWNERSHIP);
					ceres::DynamicAutoDiffCostFunction<ReprojectionResidual, 4>* reprojectionCoast =
						new ceres::DynamicAutoDiffCostFunction<ReprojectionResidual, 4>(
							new ReprojectionResidual(bilinear_weight_tar, bilinear_weight_src,
								depth_value_tar, depth_value_src,
								grid_width, grid_height));

					// separate each depth map coefficients to reduce the Jacobian matrix scale
					reprojectionCoast->AddParameterBlock(grid_width * grid_height);
					reprojectionCoast->AddParameterBlock(grid_width * grid_height);
					reprojectionCoast->AddParameterBlock(grid_width * grid_height);
					reprojectionCoast->AddParameterBlock(grid_width * grid_height);
					reprojectionCoast->SetNumResiduals(1);

					#pragma omp critical(problem)
					{
						problem.AddResidualBlock(
							reprojectionCoast,
							reprojectionLoss,
							(s_ij_list + depthmap_index_src * grid_width * grid_height),
							(o_ij_list + depthmap_index_src * grid_width * grid_height),
							(s_ij_list + depthmap_index_tar * grid_width * grid_height),
							(o_ij_list + depthmap_index_tar * grid_width * grid_height));
					}
				}
				//else if(depthmap_index_src == depthmap_ref_intidx){
				//	ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(nullptr, weight_reprojection, ceres::TAKE_OWNERSHIP);
				//	//ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(new ceres::CauchyLoss(1), weight_reprojection, ceres::TAKE_OWNERSHIP);
				//	ceres::DynamicAutoDiffCostFunction<ReprojectionResidual_fixed, 4>* reprojectionCoast =
				//		new ceres::DynamicAutoDiffCostFunction<ReprojectionResidual_fixed, 4>(
				//			new ReprojectionResidual_fixed(bilinear_weight_tar,
				//				depth_value_tar, depth_value_src, grid_width, grid_height));

				//	// separate each depth map coefficients to reduce the Jacobian matrix scale
				//	reprojectionCoast->AddParameterBlock(grid_width * grid_height);
				//	reprojectionCoast->AddParameterBlock(grid_width * grid_height);
				//	reprojectionCoast->SetNumResiduals(1);

				//	problem.AddResidualBlock(
				//		reprojectionCoast,
				//		reprojectionLoss,
				//		(s_ij_list + depthmap_index_tar * grid_width * grid_height),
				//		(o_ij_list + depthmap_index_tar * grid_width * grid_height));
				//	//problem.AddResidualBlock(
				//	//	reprojectionCoast,
				//	//	nullptr,
				//	//	(s_ij_list + depthmap_index_tar * grid_width * grid_height),
				//	//	(o_ij_list + depthmap_index_tar * grid_width * grid_height));
				//}
			}

		}// End of depthmap_counter_adjust
	}// End of depthmap_counter_ref

	// report the overlap 0 image pairs.
	if (ignore_image_pair.size() != 0)
	{
		std::stringstream ss;
		ss << "The overlap between depth map ";
		for (auto item : ignore_image_pair)
		{
			ss << item.first << "=>" << item.second << "\t";
		}
		ss << " is 0%, skip!";
		LOG(INFO) << ss.str();
	}

	// 1-2) smooth term
	LOG(INFO) << "Adding smooth term..";
	// pre-compute the edge size for smooth term
	int grid_edge_number = 4 * (grid_width - 1) * (grid_height - 1) + (grid_width - 1) + (grid_height - 1);
	int* index_current_list = (int*)malloc(grid_edge_number * sizeof(int));
	memset(index_current_list, 0, grid_edge_number * sizeof(int));
	int* index_neighbour_list = (int*)malloc(grid_edge_number * sizeof(int));
	memset(index_neighbour_list, 0, grid_edge_number * sizeof(int));

	// to be free
	bilinear_weight_list_mem.push_back((void*)index_current_list);
	bilinear_weight_list_mem.push_back((void*)index_neighbour_list);

	int edge_counter = 0;
	for (int y_index = 0; y_index < grid_height; y_index++)
	{
		for (int x_index = 0; x_index < grid_width; x_index++)
		{
			int index_current = y_index * grid_width + x_index;

			if (x_index == (grid_width - 1) && y_index == (grid_height - 1))
			{
				continue;
			}
			else if (x_index == grid_width - 1)
			{
				index_current_list[edge_counter] = (index_current);
				index_neighbour_list[edge_counter] = (index_current + grid_width);
				edge_counter++;
			}
			else if (y_index == grid_height - 1)
			{
				index_current_list[edge_counter] = (index_current);
				index_neighbour_list[edge_counter] = (index_current + 1);
				edge_counter++;
			}
			else
			{
				index_current_list[edge_counter] = (index_current);
				index_neighbour_list[edge_counter] = (index_current + 1);
				edge_counter++;
				index_current_list[edge_counter] = (index_current);
				index_neighbour_list[edge_counter] = (index_current + grid_width);
				edge_counter++;
				index_current_list[edge_counter] = (index_current);
				index_neighbour_list[edge_counter] = (index_current + grid_width + 1);
				edge_counter++;
				index_current_list[edge_counter] = (index_current + 1);
				index_neighbour_list[edge_counter] = (index_current + grid_width);
				edge_counter++;
			}
		}
	}
	if (edge_counter != grid_edge_number)
		LOG(ERROR) << "Smooth term edge number error! " << edge_counter << " , it should be " << grid_edge_number;

	for (int depthmap_index = 0; depthmap_index < depthmap_original.size(); depthmap_index++)
	{
		ceres::LossFunction* smoothnessLoss = new ceres::ScaledLoss(nullptr, weight_smooth, ceres::TAKE_OWNERSHIP);
		ceres::DynamicAutoDiffCostFunction<SmoothnessResidual, 4>* smoothnessCoast =
			new ceres::DynamicAutoDiffCostFunction<SmoothnessResidual, 4>(
				new SmoothnessResidual(index_current_list, index_neighbour_list, grid_edge_number, grid_width, grid_height));
		smoothnessCoast->AddParameterBlock(grid_width* grid_height);
		smoothnessCoast->AddParameterBlock(grid_width* grid_height);
		smoothnessCoast->SetNumResiduals(1);

		problem.AddResidualBlock(
			smoothnessCoast,
			smoothnessLoss,
			s_ij_list + depthmap_index * grid_width * grid_height,
			o_ij_list + depthmap_index * grid_width * grid_height);

		//ceres::LossFunction* smoothnessLoss_s = new ceres::ScaledLoss(nullptr, weight_smooth, ceres::TAKE_OWNERSHIP);
		//ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_S, 4>* smoothnessCoast_s =
		//	new ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_S, 4>(
		//		new SmoothnessResidual_S(index_current_list, index_neighbour_list, grid_edge_number, grid_width, grid_height));
		//smoothnessCoast_s->AddParameterBlock(grid_width* grid_height);
		//smoothnessCoast_s->AddParameterBlock(grid_width* grid_height);
		//smoothnessCoast_s->SetNumResiduals(1);

		//problem.AddResidualBlock(
		//	smoothnessCoast_s,
		//	smoothnessLoss_s,
		//	s_ij_list + depthmap_index * grid_width * grid_height,
		//	o_ij_list + depthmap_index * grid_width * grid_height);

		//ceres::LossFunction* smoothnessLoss_o = new ceres::ScaledLoss(nullptr, weight_smooth , ceres::TAKE_OWNERSHIP);
		//ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_O, 4>* smoothnessCoast_o =
		//	new ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_O, 4>(
		//		new SmoothnessResidual_O(index_current_list, index_neighbour_list, grid_edge_number, grid_width, grid_height));
		//smoothnessCoast_o->AddParameterBlock(grid_width* grid_height);
		//smoothnessCoast_o->AddParameterBlock(grid_width* grid_height);
		//smoothnessCoast_o->SetNumResiduals(1);

		//problem.AddResidualBlock(
		//	smoothnessCoast_o,
		//	smoothnessLoss_o,
		//	s_ij_list + depthmap_index * grid_width * grid_height,
		//	o_ij_list + depthmap_index * grid_width * grid_height);
	}

	//1-3) regularization term
	LOG(INFO) << "Adding regular term..";
	if (weight_scale != 0 && depthmap_ref_extidx > 0)
		LOG(WARNING) << "Both scale weight and reference frame index are set!";

	for (int depthmap_index = 0; depthmap_index < depthmap_original.size(); depthmap_index++)
	{
		ceres::LossFunction* scaleLoss = new ceres::ScaledLoss(nullptr, weight_scale, ceres::TAKE_OWNERSHIP);
		ceres::DynamicAutoDiffCostFunction<ScaleResidual, 4>* scaleCoast =
			new ceres::DynamicAutoDiffCostFunction<ScaleResidual, 4>(
				new ScaleResidual(grid_width, grid_height));
		scaleCoast->AddParameterBlock(grid_width * grid_height);
		scaleCoast->SetNumResiduals(1);
		problem.AddResidualBlock(
			scaleCoast,
			scaleLoss,
			s_ij_list + depthmap_index * grid_width * grid_height);
	}

	// fix the reference depthmap's scale and offset coefficients
	//problem.SetParameterBlockConstant(s_ij_list + depthmap_ref_intidx * grid_width * grid_height);
	//problem.SetParameterBlockConstant(o_ij_list + depthmap_ref_intidx * grid_width * grid_height);

	// 2) Solve the problem
	ceres::Solver::Options options;
	if (ceres_num_threads > 0)
	{
		LOG(INFO) << "Ceres solver num_threads is: " << ceres_num_threads;
		options.num_threads = ceres_num_threads;
	}
	if (ceres_max_num_iterations > 0)
	{
		LOG(INFO) << "Ceres solver ceres_max_num_iterations is: " << ceres_max_num_iterations;
		options.max_num_iterations = ceres_max_num_iterations;
	}
	if (ceres_max_linear_solver_iterations > 0)
	{
		LOG(INFO) << "Ceres solver ceres_max_linear_solver_iterations is: " << ceres_max_linear_solver_iterations;
		options.max_linear_solver_iterations = ceres_max_linear_solver_iterations;
	}
	if (ceres_min_linear_solver_iterations > 0)
	{
		LOG(INFO) << "Ceres solver min_linear_solver_iterations is: " << ceres_min_linear_solver_iterations;
		options.min_linear_solver_iterations = ceres_min_linear_solver_iterations;
	}

	//options.minimizer_type = ceres::TRUST_REGION;
	options.minimizer_type = ceres::LINE_SEARCH;

	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	// options.linear_solver_type = ceres::SPARSE_SCHUR; // support multi-thread

	//options.line_search_direction_type = ceres::STEEPEST_DESCENT;
	//options.line_search_direction_type = ceres::LBFGS;

	//options.line_search_type = ceres::WOLFE;

	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	std::cout << "The iteration number is: " << summary.iterations.size() << std::endl;
	std::cout << "Number inner iteration number:" << summary.num_inner_iteration_steps << std::endl;
	std::cout << "The total time consume (second): " << summary.total_time_in_seconds << std::endl;
	double time_consume_each_iter = summary.total_time_in_seconds / summary.iterations.size();
	std::cout << "Each iteration average time (second) " << time_consume_each_iter << std::endl;

	// assign & release resource
	for (void* pointer : bilinear_weight_list_mem)
		free(pointer);
}
