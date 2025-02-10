#include "depthmap_stitcher_enum.hpp"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include <vector>
#include <map>
#include <string>


DepthmapStitcherEnum::DepthmapStitcherEnum(float depthmap_optim_overlap_ratio) :
	depthmap_optim_overlap_ratio_(depthmap_optim_overlap_ratio)
{

}

//===== Ceres Energy Function Terms ========
// @see hedman2018instant:equ_1,equ_5
struct DepthmapStitcherEnum::ReprojectionResidual
{

	ReprojectionResidual(double* bilinear_weight_list, double depth_value_a, double depth_value_refernce, int grid_width, int grid_height) : bilinear_weight_list_(bilinear_weight_list), depth_value_a_(depth_value_a), depth_value_refernce_(depth_value_refernce), grid_width_(grid_width), grid_height_(grid_height) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		T scale(0);
		T offset(0);
		for (int index = 0; index < grid_width_ * grid_height_; index++)
		{
			scale += scale_offset_list[0][index] * bilinear_weight_list_[index];
			offset += scale_offset_list[1][index] * bilinear_weight_list_[index];
		}
		T temp = depth_value_refernce_ - depth_value_a_ * scale - offset;
		//T temp = depth_value_refernce_ - depth_value_a_ - offset;
		//T temp = depth_value_refernce_ - depth_value_a_ * scale;
		residual[0] = temp * temp;
		return true;
	}

private:
	const double* bilinear_weight_list_; // weight of S and A, length is grid_width_ * grid_height_
	const double depth_value_a_;		 // the depth value of source depth map
	const double depth_value_refernce_;	 // the depth value of corresponding pixel in reference depth map
	const int grid_width_;					 // the weight grid along the x axis
	const int grid_height_;					 // the weight grid along the y axis
};

// @see hedman2018instant:equ_6
struct DepthmapStitcherEnum::SmoothnessResidual
{
	SmoothnessResidual(int index_current, int index_neighbour) : index_current_(index_current), index_neighbour_(index_neighbour) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		const T* offset_list = scale_offset_list[1];

		T scale_diff = scale_list[index_current_] - scale_list[index_neighbour_];
		T offset_diff = offset_list[index_current_] - offset_list[index_neighbour_];
		residual[0] = scale_diff * scale_diff +offset_diff * offset_diff;
		return true;
	}

private:
	const int index_current_;	// the weight grid along the x axis
	const int index_neighbour_; // the weight grid along the y axis
};

// @see hedman2018instant:equ_6
struct DepthmapStitcherEnum::SmoothnessResidual_S
{
	SmoothnessResidual_S(int index_current, int index_neighbour) : index_current_(index_current), index_neighbour_(index_neighbour) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		const T* offset_list = scale_offset_list[1];

		T scale_diff = scale_list[index_current_] - scale_list[index_neighbour_];
		//T offset_diff = offset_list[index_current_] - offset_list[index_neighbour_];
		residual[0] = scale_diff * scale_diff;// +offset_diff * offset_diff;
		return true;
	}

private:
	const int index_current_;	// the weight grid along the x axis
	const int index_neighbour_; // the weight grid along the y axis
};

// @see hedman2018instant:equ_6
struct DepthmapStitcherEnum::SmoothnessResidual_O
{
	SmoothnessResidual_O(int index_current, int index_neighbour) : index_current_(index_current), index_neighbour_(index_neighbour) {}

	template <typename T>
	bool operator()(T const* const* scale_offset_list, T* residual) const
	{
		const T* scale_list = scale_offset_list[0];
		const T* offset_list = scale_offset_list[1];

		//T scale_diff = scale_list[index_current_] - scale_list[index_neighbour_];
		T offset_diff = offset_list[index_current_] - offset_list[index_neighbour_];
		residual[0] =  offset_diff * offset_diff;
		return true;
	}

private:
	const int index_current_;	// the weight grid along the x axis
	const int index_neighbour_; // the weight grid along the y axis
};

// @see hedman2018instant:equ_7
struct DepthmapStitcherEnum::ScaleResidual
{
	ScaleResidual(int grid_width, int grid_height) : grid_width_(grid_width), grid_height_(grid_height) {}

	template <typename T>
	bool operator()(T const* const* scale_list_list, T* residual) const
	{
		T sum(0);
		const T* scale_list = scale_list_list[0];
		for (int index = 0; index < grid_width_ * grid_height_; index++)
		{
			sum += 1.0 / scale_list[index];
		}
		residual[0] = sum;
		return true;
	}

private:
	const int grid_width_; // the weight grid along the x axis
	const int grid_height_; // the weight grid along the y axis
};

void DepthmapStitcherEnum::initial(const int grid_width , const int grid_height )
{
	if (depthmap_original.size() < 2)
		throw std::runtime_error("The depth map image less than 2.");

	DepthmapStitcher::initial(grid_width, grid_height);

	// the first depth as the reference depth map
	LOG(INFO) << "Initial the scale coefficients to 100 and offset to 50.";
	coeff_so.set_value_const(1.0, 0.0);

	// initial depth map vector
	depthmap_aligned.clear();
	depthmap_aligned.resize(depthmap_original.size(), cv::Mat());
}

void DepthmapStitcherEnum::compute_align_coeff()
{
	//std::cout << coeff_so << std::endl;

	// 0) load set the data and parameters
	int image_width = depthmap_original[0].cols;
	int image_height = depthmap_original[0].rows;

	// set the first image as the reference depth map & coefficients
	depthmap_aligned[0] = depthmap_original[0].clone();
	coeff_so.coeff_scale_mat[0].setTo(1.0);
	coeff_so.coeff_offset_mat[0].setTo(0.0);

	LOG(INFO) << "Terms weights are proj:" << this->weight_reprojection <<
		",\tsmooth:" << this->weight_smooth <<
		",\tscale:" << this->weight_scale;
	// 1) make the energy function,
	// adjusted depth map register to reference depth map to compute the scale and offset
	for (int depthmap_index_ref = 0; depthmap_index_ref < depthmap_original.size(); depthmap_index_ref++)
	{
		// find the adjusted depth as reference map
		if (depthmap_aligned[depthmap_index_ref].size().area() == 0)
			continue;

		// the aligned depth map as the reference depth map
		const cv::Mat depth_map_reference = depthmap_aligned[depthmap_index_ref];

		for (int depthmap_index_cur = 0; depthmap_index_cur < depthmap_original.size(); depthmap_index_cur++)
		{
			// skip the adjusted depth map
			if (depthmap_aligned[depthmap_index_cur].size().area() > 0)
				continue;

			LOG(INFO) << "Reference depth map " << depthmap_index_ref << " and current depth map " << depthmap_index_cur;
			const cv::Mat depth_map_current = depthmap_original[depthmap_index_cur];

			// adjusted depth map scale and offset
			double* s_ij = reinterpret_cast<double*>(coeff_so.coeff_scale_mat[depthmap_index_cur].data);
			double* o_ij = reinterpret_cast<double*>(coeff_so.coeff_offset_mat[depthmap_index_cur].data);

			// pixels corresponding relationship
			const cv::Mat pixels_corresponding = pixels_corresponding_list.at(depthmap_index_cur).at(depthmap_index_ref);
			int observation_pairs_number = pixels_corresponding.rows;

			if (observation_pairs_number < (depthmap_optim_overlap_ratio_ * depth_map_current.size().area()))
			{
				float overlap_ratio = (double)observation_pairs_number / (double)depth_map_current.size().area();
				LOG(INFO) << "The overlap between depth map " << depthmap_index_ref << " and depth map " << depthmap_index_cur << " is " << overlap_ratio * 100.0 << " which is less than " << depthmap_optim_overlap_ratio_ * 100 << "%, skip the optimization!";
				continue;
			}

			ceres::Problem problem;
			// 1-1) re-projection term
			// grid interpolation weight for each pixel, row-major
			double* bilinear_weight_list = (double*)malloc(grid_width * grid_height * observation_pairs_number * sizeof(double));
			memset(bilinear_weight_list, 0, grid_width * grid_height * observation_pairs_number * sizeof(double));

			// add the depth value and weight value
			for (int observations_index = 0; observations_index < observation_pairs_number; observations_index++)
			{
				const double y_cur = pixels_corresponding.at<double>(observations_index, 0);
				const double x_cur = pixels_corresponding.at<double>(observations_index, 1);
				const double y_ref = pixels_corresponding.at<double>(observations_index, 2);
				const double x_ref = pixels_corresponding.at<double>(observations_index, 3);

				double depth_value_current = getColorSubpix(depth_map_current, cv::Point2f(x_cur, y_cur));
				double depth_value_refernce = getColorSubpix(depth_map_reference, cv::Point2f(x_ref, y_ref));

				// compute the bilinear weights;
				double* bilinear_weight = bilinear_weight_list + grid_width * grid_height * observations_index;
				get_bilinear_weight(bilinear_weight, image_width, image_height, grid_height, grid_width, x_cur, y_cur);
				//cv::Mat bilinear_weight_list_mat(cv::Size(image_width, image_height), CV_64FC1, bilinear_weight);

				// add residual block
				//ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(new ceres::CauchyLoss(1), weight_reprojection, ceres::TAKE_OWNERSHIP);
				ceres::LossFunction* reprojectionLoss = new ceres::ScaledLoss(nullptr, weight_reprojection, ceres::TAKE_OWNERSHIP);
				ceres::DynamicAutoDiffCostFunction<ReprojectionResidual, 4>* reprojectionCoast =
					new ceres::DynamicAutoDiffCostFunction<ReprojectionResidual, 4>(
						new ReprojectionResidual(bilinear_weight, depth_value_current, depth_value_refernce, grid_width, grid_height));
				reprojectionCoast->AddParameterBlock(grid_width * grid_height);
				reprojectionCoast->AddParameterBlock(grid_width * grid_height);
				reprojectionCoast->SetNumResiduals(1);

				problem.AddResidualBlock(reprojectionCoast,
					reprojectionLoss,
					s_ij,
					o_ij);
			}

			 //1-2) smooth term
			std::vector<int> index_current_list;
			std::vector<int> index_neighbour_list;
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
						index_current_list.push_back(index_current);
						index_neighbour_list.push_back(index_current + grid_width);
					}
					else if (y_index == grid_height - 1)
					{
						index_current_list.push_back(index_current);
						index_neighbour_list.push_back(index_current + 1);
					}
					else
					{
						index_current_list.push_back(index_current);
						index_neighbour_list.push_back(index_current + 1);
						index_current_list.push_back(index_current);
						index_neighbour_list.push_back(index_current + grid_width);
						index_current_list.push_back(index_current);
						index_neighbour_list.push_back(index_current + grid_width + 1);
						index_current_list.push_back(index_current + 1);
						index_neighbour_list.push_back(index_current + grid_width);
					}
				}
			}
			for (int edge_index = 0; edge_index < index_current_list.size(); edge_index++)
			{
				int index_current = index_current_list[edge_index];
				int index_neighbour = index_neighbour_list[edge_index];
				//std::cout << "current edge index:" << index_current << ", neighbour edge index:" << index_neighbour << std::endl;

				ceres::LossFunction* smoothnessloss_s = new ceres::ScaledLoss(nullptr, weight_smooth, ceres::TAKE_OWNERSHIP);
				ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_S, 4>* smoothnesscoast_s =
					new ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_S, 4>(
						new SmoothnessResidual_S(index_current, index_neighbour));
				smoothnesscoast_s->AddParameterBlock(grid_width * grid_height);
				smoothnesscoast_s->AddParameterBlock(grid_width * grid_height);
				smoothnesscoast_s->SetNumResiduals(1);

				problem.AddResidualBlock(
					smoothnesscoast_s,
					smoothnessloss_s,
					s_ij,
					o_ij);

				ceres::LossFunction* smoothnessloss_o = new ceres::ScaledLoss(nullptr, weight_smooth, ceres::TAKE_OWNERSHIP);
				ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_O, 4>* smoothnesscoast_o =
					new ceres::DynamicAutoDiffCostFunction<SmoothnessResidual_O, 4>(
						new SmoothnessResidual_O(index_current, index_neighbour));
				smoothnesscoast_o->AddParameterBlock(grid_width * grid_height);
				smoothnesscoast_o->AddParameterBlock(grid_width * grid_height);
				smoothnesscoast_o->SetNumResiduals(1);

				problem.AddResidualBlock(
					smoothnesscoast_o,
					smoothnessloss_o,
					s_ij,
					o_ij);

				//ceres::lossfunction* smoothnessloss = new ceres::scaledloss(nullptr, weight_smooth, ceres::take_ownership);
				//ceres::dynamicautodiffcostfunction<smoothnessresidual, 4>* smoothnesscoast =
				//	new ceres::dynamicautodiffcostfunction<smoothnessresidual, 4>(
				//		new smoothnessresidual(index_current, index_neighbour));
				//smoothnesscoast->addparameterblock(grid_x * grid_y);
				//smoothnesscoast->addparameterblock(grid_x * grid_y);
				//smoothnesscoast->setnumresiduals(1);

				//problem.addresidualblock(
				//	smoothnesscoast,
				//	smoothnessloss,
				//	s_ij,
				//	o_ij);
			}

			//1-3) regularization term
			if (weight_scale != 0)
				LOG(WARNING) << "Scale Term weight is not 0.0.";
			ceres::LossFunction* scaleLoss = new ceres::ScaledLoss(nullptr, weight_scale, ceres::TAKE_OWNERSHIP);
			ceres::DynamicAutoDiffCostFunction<ScaleResidual, 4>* scaleCoast =
				new ceres::DynamicAutoDiffCostFunction<ScaleResidual, 4>(
					new ScaleResidual(grid_width, grid_height));
			scaleCoast->AddParameterBlock(grid_width* grid_height);
			scaleCoast->SetNumResiduals(1);

			problem.AddResidualBlock(
				scaleCoast,
				scaleLoss,
				s_ij);

			// Solve the problem
			ceres::Solver::Options options;
			if (ceres_num_threads > 0)
				options.num_threads = ceres_num_threads;
			if (ceres_max_num_iterations > 0)
				options.max_num_iterations = ceres_max_num_iterations;
			if (ceres_max_linear_solver_iterations > 0)
				options.max_linear_solver_iterations = ceres_max_linear_solver_iterations;
			if (ceres_min_linear_solver_iterations > 0)
				options.min_linear_solver_iterations = ceres_min_linear_solver_iterations;

			//options.linear_solver_type = ceres::CGNR;
			options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
			//options.minimizer_type = ceres::LINE_SEARCH;
			options.minimizer_progress_to_stdout = true;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.FullReport() << "\n";

			// release resource
			free(bilinear_weight_list);
			bilinear_weight_list = nullptr;

			// 2) adjust depth map and update the data
			align_depthmap(depthmap_index_cur);

		} // End of depthmap_counter_dest
	} // End of depthmap_counter_src
}
