#pragma once

#include "depthmap_stitcher.hpp"

namespace cv
{
	class Mat;
}

class DepthmapStitcherEnum : public DepthmapStitcher
{

public:
	DepthmapStitcherEnum(float depthmap_optim_overlap_ratio = 0.25);

	~DepthmapStitcherEnum() {};

	/**
	 * Get the grid size .
	 * The sub-class should call this super-call function.
	 */
	virtual void initial(const int grid_width, const int grid_height) override;

	/**
	 * Compute each depth maps S and A relative reference depth map.
	 * Use the first depth map in depth_map_list as the reference map.
	 * And enumerate all image pairs to compute the scale and offset.
	 */
	void compute_align_coeff() override;

	// the overlap area ratio, the selected images pairs overlap area is larger than it.
	float depthmap_optim_overlap_ratio_;

private:

	struct ReprojectionResidual;

	struct SmoothnessResidual_S;
	struct SmoothnessResidual_O;
	struct SmoothnessResidual;
	struct ScaleResidual;
};