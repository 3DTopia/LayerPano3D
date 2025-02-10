#pragma once

#include "depthmap_stitcher.hpp"

namespace cv
{
	class Mat;
}

class DepthmapStitcherGroup : public DepthmapStitcher
{

public:
	DepthmapStitcherGroup();

	~DepthmapStitcherGroup();

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

private:

	struct ReprojectionResidual;
	struct ReprojectionResidual_fixed;

	struct SmoothnessResidual;
	struct SmoothnessResidual_O;
	struct SmoothnessResidual_S;

	struct ScaleResidual;
};