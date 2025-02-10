import pickle
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
# import open3d as o3d
import torch
import sys
import shutil


python_src_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(python_src_dir)

print(f"Adding '{parent_dir}' to sys.path") 
sys.path.append(parent_dir + "/submodules/360monodepth/code/python/src/") 



from utility import blending, image_io, depthmap_utils, serialization, pointcloud_utils, subimage
from utility.logger import Logger
from utility.projection_icosahedron import erp2ico_image, ico2erp_image

log = Logger(__name__)
log.logger.propagate = False


try:
    from instaOmniDepth import depthmapAlign
    log.info("depthmapAlign python module installed!")
except ModuleNotFoundError:
    log.error("depthmapAlign python module do not install, please build and install it reference the readme.")
 

def scale_shift_linear_disp(rendered_depth, predicted_depth, mask, fuse=True):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

    :param rendered_depth: torch.Tensor (H, W)
    :param predicted_depth:  torch.Tensor (H, W)
    :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

    :return: scale/shift corrected depth
    """
    if mask.sum() == 0:
        return predicted_depth

    rendered_disparity = 1 / rendered_depth[mask].unsqueeze(-1)
    predicted_disparity = 1 / predicted_depth[mask].unsqueeze(-1)

    X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ rendered_disparity
    AB = XTX_inv @ XTY

    error = rendered_disparity - X @ AB


    fixed_disparity = (1 / predicted_depth) * AB[0] + AB[1]
    fixed_depth = 1 / fixed_disparity

    if fuse:
        fused_depth = torch.where(mask, rendered_depth, fixed_depth)
        return fused_depth
    else:
        return fixed_depth




def scale_shift_linear(rendered_depth, predicted_depth, max_val, mask, fuse=True):
    """
    Optimize a scale and shift parameter in the least squares sense, such that rendered_depth and predicted_depth match.
    Formally, solves the following objective:

    min     || (d * a + b) - d_hat ||
    a, b

    where d = 1 / predicted_depth, d_hat = 1 / rendered_depth

    :param rendered_depth: torch.Tensor (H, W)
    :param predicted_depth:  torch.Tensor (H, W)
    :param mask: torch.Tensor (H, W) - 1: valid points of rendered_depth, 0: invalid points of rendered_depth (ignore)
    :param fuse: whether to fuse shifted/scaled predicted_depth with the rendered_depth

    :return: scale/shift corrected depth
    """

    predicted_depth_raw = predicted_depth.clone()
    h, w = predicted_depth.shape
    rendered_depth = rendered_depth[mask].unsqueeze(-1).float()
    predicted_depth = predicted_depth[mask].unsqueeze(-1).float()
    X = torch.cat([predicted_depth, torch.ones_like(predicted_depth)], dim=1)
    XTX_inv = (X.T @ X).inverse()
    XTY = X.T @ rendered_depth
    AB = XTX_inv @ XTY
    fixed_depth = predicted_depth_raw * AB[0] + AB[1]
    error = rendered_depth - X @ AB



    # h, w = predicted_depth.shape
    # rendered_disparity = 1 / rendered_depth[mask].unsqueeze(-1)
    # predicted_disparity = 1 / predicted_depth[mask].unsqueeze(-1)

    # X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    # XTX_inv = (X.T @ X).inverse()
    # XTY = X.T @ rendered_disparity
    # AB = XTX_inv @ XTY

    # fixed_disparity = (1 / predicted_depth) * AB[0] + AB[1]
    # fixed_depth = 1 / fixed_disparity



    # h, w = predicted_depth.shape
    # rendered_disparity = max_val - rendered_depth
    # rendered_disparity = rendered_disparity[mask].unsqueeze(-1)
    # predicted_disparity = max_val - predicted_depth
    # predicted_disparity = predicted_disparity[mask].unsqueeze(-1)
    # X = torch.cat([predicted_disparity, torch.ones_like(predicted_disparity)], dim=1)
    # XTX_inv = (X.T @ X).inverse()
    # XTY = X.T @ rendered_disparity
    # AB = XTX_inv @ XTY
    # fixed_disparity = (max_val - predicted_depth) * AB[0] + AB[1]
    # fixed_depth = max_val - fixed_disparity


    return fixed_depth.view(h,w)


def depth_alignment(depth1, depth2, mask_strict):
    # modify depth2 to match depth1, 
    # depth2_fixed = depth alignment(depth_next, depth_current, mask)
    
    # depth1    #rendered_depth
    # depth2    #predicted_depth

    depth1 = depth1.astype(np.int32)
    depth2 = depth2.astype(np.int32)
    

    if len(depth2.shape) == 3:
        depth2 = depth2[:,:,1]


    mask4align = mask_strict.astype(bool) 
    max_val = depth2.max().astype(float)
    
    # max_val = 65535

    depth1 = depth1.astype(depth2.dtype)
    depth2_fixed = scale_shift_linear(torch.from_numpy(depth1), torch.from_numpy(depth2), \
                                      max_val.astype(float), torch.from_numpy(mask4align), fuse=True)
    
    # depth2_fixed = align_depth_maps_lsq(depth1, depth2 , mask4align)   

    if max_val < 300:
        depth2_fixed = depth2_fixed.detach().cpu().numpy().astype(np.uint8)
    else:
        depth2_fixed = depth2_fixed.detach().cpu().numpy().astype(np.int32)
    return depth2_fixed

def align_depth_maps_lsq(render_depth, predict_depth, mask):



    y_indices, x_indices = np.where(mask)
    common_render_depth = render_depth[mask]
    common_predict_depth = predict_depth[mask]
    
    A = np.vstack([x_indices, y_indices, np.ones_like(x_indices)]).T
    b = common_predict_depth - common_render_depth
    
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    M = np.array([
        [1, 0, params[0]],
        [0, 1, params[1]],
        [0, 0, 1]
    ])
    
    h, w = predict_depth.shape
    aligned_predict_depth = cv2.warpAffine(predict_depth, M[:2], (w, h), flags=cv2.INTER_LINEAR)
    aligned_predict_depth = torch.from_numpy(aligned_predict_depth)

    return aligned_predict_depth



class DepthmapAlign:
    """The class wrap cpp module.

    The cpp python module function declaration:
    1. `depthmap_stitch` align subimage's depth maps together, parameters list:
        str: root_dir,
        str: method,
        list: terms_weight,
        list: depthmap_original_list,
        list: depthmap_original_ico_index,
        int: reference_depthmap_index, 
        list: pixels_corresponding_map,
        int: align_coeff_grid_height,
        int: align_coeff_grid_width,
        list: align_coeff_initial_scale,
        list: align_coeff_initial_offset.

    """

    def __init__(self, opt, output_dir, subimages_rgb, debug=False):
        # sub-image number
        self.depthmap_number = -1
        self.output_dir = output_dir  # output alignment coefficient
        self.depthmap_aligned = None

        # output debug information filepath exp
        self.debug = debug
        # if output path is None do not output the data to files
        self.subimage_pixelcorr_filepath_expression = None
        self.subimage_depthmap_aligning_filepath_expression = None  # save the normalized depth map for each resolution
        self.subimage_warpedimage_filepath_expression = None
        self.subimage_warpeddepth_filename_expression = None
        self.subimage_alignment_intermedia_filepath_expression = None   # pickle file
        self.subimages_rgb = subimages_rgb  # the original 20 rgb subimages, if is not warp depth map

        # align grid
        self.align_coeff_grid_width = 8           # the width of the initial grid
        self.align_coeff_grid_height = 7          # the height of the initial grid
        self.align_coeff_grid_width_finest = 10   # the grid width of the finest grid
        self.align_coeff_grid_height_finest = 16  # the grid height of the finest grid.
        self.align_coeff_initial_scale_list = []
        self.align_coeff_initial_offset_list = []
        self.depthmap_original_ico_index = []      # the subimage's depth map ico face index

        # ceres options
        self.ceres_thread_number = 12
        self.ceres_max_num_iterations = 100 
        self.ceres_max_linear_solver_iterations = 10
        self.ceres_min_linear_solver_iterations = -1

        # align parameter
        self.align_method = "group"
        self.weight_project = 1.0
        self.weight_smooth = 40 
        self.weight_scale = 0.007
        self.depthmap_norm_mothod = "midas"
        self.coeff_fixed_face_index = -1 

        # multi-resolution parameters
        self.multi_res_grid = False
        self.pyramid_layer_number = 1
        self.pyramid_downscale = 2

        # pixel correpsonding down-sample parameter
        self.downsample_pixelcorr_ratio = 0.001

        # initial the align process (depthmapAlign) run time
        depthmapAlign.init(self.align_method)
        # clear the alignment run time, call depthmapAlign.shutdown when the interpreter exits
        import atexit
        atexit.register(depthmapAlign.shutdown)

        # the global configuration
        self.opt = opt 

    def align_coeff_init(self):
        """
        Create & initial subimages alignment coefficient.
        """
        for ico_face_index in self.depthmap_original_ico_index:
            align_coeff_initial_scale = np.full((self.align_coeff_grid_height, self.align_coeff_grid_width), 1.0, np.float64)
            align_coeff_initial_offset = np.full((self.align_coeff_grid_height, self.align_coeff_grid_width), 0.0, np.float64)
            self.align_coeff_initial_scale_list.append(align_coeff_initial_scale)
            self.align_coeff_initial_offset_list.append(align_coeff_initial_offset)

    def report_cost(self, depthmap_list, pixel_corr_list):
        """ Report the cost of depth map alignment.

        :param depthmap_list: the depth map lists
        :type depthmap_list: list
        :param pixel_corr_list: the pixel corresponding relationship between two subimage.
        :type pixel_corr_list: dict
        """
        diff_sum = 0
        pixel_numb = 0
        # the cost of projection term
        for src_idx in range(0,20):
            for tar_idx in range(0,20):
                if src_idx == tar_idx:
                    continue

                pixel_corr = pixel_corr_list[src_idx][tar_idx]
                if pixel_corr.size == 0:
                    continue

                src_depthmap = depthmap_list[src_idx]
                tar_depthmap = depthmap_list[tar_idx]

                src_y = pixel_corr[:,0]
                src_x = pixel_corr[:,1]
                tar_y = pixel_corr[:,2]
                tar_x = pixel_corr[:,3]

                from scipy import ndimage
                src_depthmap_points = ndimage.map_coordinates(src_depthmap, [src_y, src_x], order=1, mode='constant', cval=0)
                tar_depthmap_points = ndimage.map_coordinates(tar_depthmap, [tar_y, tar_x], order=1, mode='constant', cval=0)

                diff = src_depthmap_points - tar_depthmap_points
                diff_sum += np.sum(diff * diff)
                pixel_numb += pixel_corr.shape[0]

        # print("Re-projection cost is {}, per pixel is {}".format(diff_sum, diff_sum / pixel_numb))

        # the cost of smooth term
        # the cost of scale term

    def align_single_res(self, depthmap_original_list, pixels_corresponding_list):
        """
        Align the sub-images depth map in single layer.

        :param depthmap_original_list: the not alignment depth maps.
        :type depthmap_original_list: list
        :param pixels_corresponding_list: the pixels corresponding relationship.
        :type pixels_corresponding_list: 
        """
        if self.align_method not in ["group", "enum"]:
            log.error("The depth map alignment method {} specify error! ".format(self.align_method))

        # report the alignment information:
        if False:
            # 0) get how many pixel corresponding relationship
            pixels_corr_number = 0
            info_str = ""
            for src_key, _ in enumerate(pixels_corresponding_list):
                # print("source image {}:".format(src_key))
                info_str = info_str + "\nsource image {}:\n".format(src_key)
                for tar_key, _ in enumerate(pixels_corresponding_list[src_key]):
                    # print("\t Target image {} , pixels corr number is {}".format(tar_key, pixels_corresponding_list[src_key][tar_key].size))
                    info_str = info_str + "{}:{}  ".format(tar_key, pixels_corresponding_list[src_key][tar_key].size)
                    pixels_corr_number = pixels_corr_number + pixels_corresponding_list[src_key][tar_key].size
            print(info_str)
            print("The total pixel corresponding is {}".format(pixels_corr_number))

            # 1) get the cost
            self.report_cost(depthmap_original_list, pixels_corresponding_list)

        try:
            # set Ceres solver options
            ceres_setting_result = depthmapAlign.ceres_solver_option(self.ceres_thread_number,  self.ceres_max_num_iterations,
                                                                     self.ceres_max_linear_solver_iterations, self.ceres_min_linear_solver_iterations)

            if ceres_setting_result < 0:
                log.error("Ceres solver option setting error.")

            # align depth maps
            cpp_module_debug_flag = 1 if self.debug else 0

            # align the subimage's depth maps
            self.depthmap_aligned, align_coeff = depthmapAlign.depthmap_stitch(
                self.output_dir,
                [self.weight_project, self.weight_smooth, self.weight_scale],
                depthmap_original_list,
                self.depthmap_original_ico_index,
                self.coeff_fixed_face_index,
                pixels_corresponding_list,
                self.align_coeff_grid_height,
                self.align_coeff_grid_width,
                True,
                True,
                self.align_coeff_initial_scale_list,
                self.align_coeff_initial_offset_list,
                False)

            ## report the error between the aligned depth maps
            # depthmapAlign.report_aligned_depthmap_error()

        except RuntimeError as error:
            log.error('Error: ' + repr(error))

        # update the coeff
        for index in range(0, self.depthmap_number):
            assert self.align_coeff_initial_scale_list[index].shape == align_coeff[index * 2].shape
            assert self.align_coeff_initial_offset_list[index].shape == align_coeff[index * 2 + 1].shape
            self.align_coeff_initial_scale_list[index] = align_coeff[index * 2]
            self.align_coeff_initial_offset_list[index] = align_coeff[index * 2 + 1]

    def align_multi_res(self, erp_rgb_image_data, subimage_depthmap, padding_size, depthmap_original_ico_index=None):
        """
        Align the sub-images depth map in multi-resolution.

        :param erp_rgb_image_data: the erp image used to compute the pixel corresponding relationship.
        :type erp_rgb_image_data: numpy
        :param subimage_depthmap: The sub-images depth map, generated by MiDaS.
        :type subimage_depthmap: list[numpy]
        :param padding_size: the padding size
        :type padding_size: float
        :param subsample_corr_factor: the pixel corresponding subimage factor
        :type subsample_corr_factor: float
        :param depthmap_original_ico_index: the subimage depth map's index.
        :type depthmap_original_ico_index: list
        :return: aligned depth map and coefficient.
        :rtype: tuple
        """
        self.depthmap_number = len(subimage_depthmap)

        if depthmap_original_ico_index is None and len(subimage_depthmap) == 20:
            self.depthmap_original_ico_index = list(range(0, 20))
        elif depthmap_original_ico_index is not None and len(depthmap_original_ico_index) == len(subimage_depthmap):
            self.depthmap_original_ico_index = depthmap_original_ico_index
        else:
            log.error("Do not set the ico face index.")

        # normalize the data
        log.debug("Normalization the depth map with {} norm method".format(self.depthmap_norm_mothod))
        subimage_depthmap_norm_list = []
        for depthmap in subimage_depthmap:
            subimage_depthmap_norm = depthmap_utils.dispmap_normalize(depthmap, self.depthmap_norm_mothod)
            subimage_depthmap_norm_list.append(subimage_depthmap_norm)

        # 0) generate the gaussion pyramid of each sub-image depth map
        # the 1st list is lowest resolution
        if self.multi_res_grid:
            depthmap_pryamid = [subimage_depthmap_norm_list] * self.pyramid_layer_number
            # pyramid_grid = [[4, 3], [8, 7], [16, 14]]   # Values reported in paper
            pyramid_grid = [[self.align_coeff_grid_width*(2**level),
                             self.align_coeff_grid_height*(2**level)] for
                            level in range(0, self.pyramid_layer_number)]
        else:
            depthmap_pryamid = depthmap_utils.depthmap_pyramid(subimage_depthmap_norm_list, self.pyramid_layer_number, self.pyramid_downscale)

        # 1) multi-resolution to compute the alignment coefficient
        subimage_cam_param_list = None
        for pyramid_layer_index in range(0, self.pyramid_layer_number):
            if pyramid_layer_index == 0:
                if self.multi_res_grid:
                    self.align_coeff_grid_width = pyramid_grid[pyramid_layer_index][0]
                    self.align_coeff_grid_height = pyramid_grid[pyramid_layer_index][1]
                self.align_coeff_init()

            log.info("Aligen the depth map in resolution {}".format(depthmap_pryamid[pyramid_layer_index][0].shape))
            tangent_image_width = depthmap_pryamid[pyramid_layer_index][0].shape[1]

            pixel_corr_list = None
            subimage_cam_param_list = None
            # load the corresponding relationship from file
            if self.debug:
                tangent_image_height = depthmap_pryamid[pyramid_layer_index][0].shape[0]
                image_size_str = "{}x{}".format(tangent_image_height, tangent_image_width)

                # load depthmap and relationship from pickle for fast debug
                if self.subimage_alignment_intermedia_filepath_expression is not None:
                    pickle_file_path = self.subimage_alignment_intermedia_filepath_expression.format(image_size_str)
                    if os.path.exists(pickle_file_path) and os.path.getsize(pickle_file_path) > 0:
                        log.warn("Load depthmap alignment data from {}".format(pickle_file_path))
                        alignment_data = None
                        with open(pickle_file_path, 'rb') as file:
                            alignment_data = pickle.load(file)

                        subimage_depthmap = alignment_data["subimage_depthmap"]
                        depthmap_original_ico_index = alignment_data["depthmap_original_ico_index"]
                        pixel_corr_list = alignment_data["pixel_corr_list"]
                        subimage_cam_param_list = alignment_data["subimage_cam_param_list"]

            # 1-0) get subimage the pixel corresponding relationship
            if pixel_corr_list is None or subimage_cam_param_list is None:
                _, subimage_cam_param_list, pixel_corr_list = \
                    subimage.erp_ico_proj(erp_rgb_image_data, 
                                          padding_size, 
                                          tangent_image_width, 
                                          self.downsample_pixelcorr_ratio, 
                                          self.opt)
            
            # save intermedia data for debug output pixel corresponding relationship and warped source image
            if self.debug:
                tangent_image_height = depthmap_pryamid[pyramid_layer_index][0].shape[0]
                image_size_str = "{}x{}".format(tangent_image_height, tangent_image_width)

                # save depth map and relationship and etc. to pickle for debug
                if self.subimage_alignment_intermedia_filepath_expression is not None:
                    pickle_file_path = self.subimage_alignment_intermedia_filepath_expression.format(image_size_str)
                    with open(pickle_file_path, 'wb') as file:
                        pickle.dump({"subimage_depthmap": subimage_depthmap,
                                     "depthmap_original_ico_index": depthmap_original_ico_index,
                                     "pixel_corr_list": pixel_corr_list,
                                     "subimage_cam_param_list": subimage_cam_param_list}, file)
                        log.warn("Save depth map alignment data to {}".format(pickle_file_path))

                # output the all subimages depth map corresponding relationship to json
                if self.subimage_pixelcorr_filepath_expression is not None:
                    log.debug("output the all subimages corresponding relationship to {}".format(self.subimage_pixelcorr_filepath_expression))
                    for subimage_index_src in range(0, 20):
                        for subimage_index_tar in range(0, 20):
                            if subimage_index_src == subimage_index_tar:
                                continue

                            pixel_corresponding = pixel_corr_list[subimage_index_src][subimage_index_tar]
                            json_file_path = self.subimage_pixelcorr_filepath_expression \
                                .format(subimage_index_src, subimage_index_tar, image_size_str)
                            serialization.pixel_corresponding_save(
                                json_file_path, str(subimage_index_src), None,
                                str(subimage_index_tar), None, pixel_corresponding)

                # draw the corresponding relationship in available subimage rgb images
                if self.subimage_warpedimage_filepath_expression is not None and self.subimages_rgb is not None:
                    log.debug("draw the corresponding relationship in subimage rgb and output to {}".format(self.subimage_warpedimage_filepath_expression))
                    for index_src in range(len(depthmap_original_ico_index)):
                        for index_tar in range(len(depthmap_original_ico_index)):
                            # draw relationship in rgb images
                            face_index_src = depthmap_original_ico_index[index_src]
                            face_index_tar = depthmap_original_ico_index[index_tar]
                            pixel_corresponding = pixel_corr_list[face_index_src][face_index_tar]
                            src_image_rgb = self.subimages_rgb[face_index_src]
                            tar_image_rgb = self.subimages_rgb[face_index_tar]
                            _, _, src_warp = subimage.draw_corresponding(src_image_rgb, tar_image_rgb, pixel_corresponding)
                            warp_image_filepath = self.subimage_warpedimage_filepath_expression \
                                .format(face_index_src, face_index_tar, image_size_str)
                            image_io.image_save(src_warp, warp_image_filepath)

                # draw the corresponding relationship in available subimage depth maps
                if self.subimage_warpeddepth_filename_expression is not None:
                    log.debug("draw the corresponding relationship in subimage depth map and output to {}".format(self.subimage_warpeddepth_filename_expression))
                    for index_src in range(len(depthmap_original_ico_index)):
                        for index_tar in range(len(depthmap_original_ico_index)):
                            src_image_data = depthmap_pryamid[pyramid_layer_index][index_src]
                            tar_image_data = depthmap_pryamid[pyramid_layer_index][index_tar]
                            # visualize depth map
                            src_image_rgb = depthmap_utils.depth_visual(src_image_data)
                            tar_image_rgb = depthmap_utils.depth_visual(tar_image_data)
                            # draw relationship
                            face_index_src = depthmap_original_ico_index[index_src]
                            face_index_tar = depthmap_original_ico_index[index_tar]
                            pixel_corresponding = pixel_corr_list[face_index_src][face_index_tar]
                            _, _, src_warp = subimage.draw_corresponding(src_image_rgb, tar_image_rgb, pixel_corresponding)
                            warp_image_filepath = self.subimage_warpeddepth_filename_expression \
                                .format(face_index_src, face_index_tar, image_size_str)
                            image_io.image_save(src_warp, warp_image_filepath)

                # output input depth map of each subimage
                if self.subimage_depthmap_aligning_filepath_expression is not None:
                    log.debug("output subimage's depth map of multi-layers: layer {}".format(pyramid_layer_index))
                    for index in range(len(depthmap_original_ico_index)):
                        image_data = depthmap_pryamid[pyramid_layer_index][index]
                        face_index = depthmap_original_ico_index[index]
                        subimage_depthmap_filepath = self.subimage_depthmap_aligning_filepath_expression.format(face_index, image_size_str)
                        depthmap_utils.depth_visual_save(image_data, subimage_depthmap_filepath)

            # 1-1) align depth maps, to update align coeffs and subimages depth maps.
            if self.multi_res_grid:
                if self.depthmap_aligned is not None:
                    self.align_single_res(self.depthmap_aligned, pixel_corr_list)
                else:
                    self.align_single_res(depthmap_pryamid[pyramid_layer_index], pixel_corr_list)
            else:
                self.align_single_res(depthmap_pryamid[pyramid_layer_index], pixel_corr_list)

            if self.multi_res_grid:
                if pyramid_layer_index < self.pyramid_layer_number - 1:
                    self.align_coeff_grid_width = pyramid_grid[pyramid_layer_index + 1][0]
                    self.align_coeff_grid_height = pyramid_grid[pyramid_layer_index + 1][1]
                    for i in range(0, len(self.align_coeff_initial_scale_list)):
                        self.align_coeff_initial_scale_list[i] = cv2.resize(self.align_coeff_initial_scale_list[i],
                                                                            dsize=pyramid_grid[pyramid_layer_index + 1],
                                                                            interpolation=cv2.INTER_LINEAR)

                        self.align_coeff_initial_offset_list[i] = cv2.resize(self.align_coeff_initial_offset_list[i],
                                                                             dsize=pyramid_grid[pyramid_layer_index + 1],
                                                                             interpolation=cv2.INTER_LINEAR)

        # 2) return alignment coefficients and aligned depth maps
        return self.depthmap_aligned, \
               self.align_coeff_initial_scale_list, self.align_coeff_initial_offset_list, \
               subimage_cam_param_list



########## pano depth estimation #########

class Pano_depth_estimation:
    def __init__(self, pano_H, pano_W, save_dir=None, device='cuda', depth_model='DepthAnythingv2'):
        
        
        pers_img_size = int((pano_H /1024) * 512)
        
        self.save_dir = save_dir
        self.depth_align_dir = os.path.join(self.save_dir, "depth_align/")
        self.depthmap_norm_mothod = 'midas' #{naive, midas, range01}
        os.makedirs(self.depth_align_dir, exist_ok=True) 
        self.device = device
        
        self.persp_monodepth = depth_model         #['midas3', 'DepthAnythingv2'，’zoedepth‘]
        self.tangent_img_width = pers_img_size
        self.subimage_padding_size = 0.3

        # stage 1: panorama_to_tangent_images
        self.subimg_rgb_list = [] 
        self.subimg_gnomo_xy = []  # to convert the perspective image to ERP image
        # stage 2: depth estimation
        self.depthmap_persp_list = []
        self.depthmap_erp_list = []  # the depth map in ERP image space
        self.dispmap_erp_list = []
        self.dispmap_persp_list = []
        # stage 3: depth alignment
        self.dispmap_aligned_list = [] 
        self.subimg_cam_list = [] 

        
        opt_dict = {"pano_height": pano_H        ,
                    "pano_width":  pano_W        ,
                    "subimage_padding_size": 0.3 ,
                    "tangent_img_width": pers_img_size  ,
                    "persp_monodepth": self.persp_monodepth   ,
                    "dataset_matterport_hexagon_mask_enable": False ,
                    "dataset_matterport_blurarea_shape": "circle"   ,
                    "dataset_matterport_blur_area_height": 0        ,
                    "dispalign_corr_thread_number": 10              ,
                    "blending_method": 'frustum'    #['poisson', 'frustum', 'radial', 'nn', 'mean', 'all']
                    }
        self.opt = OmegaConf.create(opt_dict)           
        
    def get_panodepth(self, pano_rgb):
        ### pano_rgb: [h,w,3] [0-255] uint8
        
        self.pano_rgb = pano_rgb
        
        self.panorama_to_tangent_images()
        self.depth_estimation()
        pano_depth = self.depth_alignment()         
        
        return pano_depth
    
    @torch.no_grad()
    def panorama_to_tangent_images(self): 
        # load panorama image 

        ### pano: [h,w,3] [0-255] uint8
        # projection to tangent images
        self.subimg_rgb_list, _, points_gnomocoord = erp2ico_image(
            self.pano_rgb, self.opt.tangent_img_width, 
            padding_size=self.opt.subimage_padding_size, 
            full_face_image=True
            )
        self.subimg_gnomo_xy = points_gnomocoord[1]


    @torch.no_grad()
    def depth_estimation(self): 
        # estimate disparity map
        self.dispmap_persp_list = depthmap_utils.run_persp_monodepth(self.subimg_rgb_list, self.opt.persp_monodepth)

        
        erp_pred = depthmap_utils.run_persp_monodepth([self.pano_rgb], self.opt.persp_monodepth)[0]
        self.erp_mask = np.abs((erp_pred -  erp_pred.min()) / (erp_pred.max() - erp_pred.min())) > 0
        
        self.no_zeros_index = np.where(self.erp_mask != 0)
        # convert disparity map to depth map 
        for dispmap_persp in self.dispmap_persp_list:
            if self.persp_monodepth != 'zoedepth':
                depthmap_persp = depthmap_utils.disparity2depth(dispmap_persp)   
            else:
                depthmap_persp = dispmap_persp

            depthmap_erp = depthmap_utils.subdepthmap_tang2erp(depthmap_persp, self.subimg_gnomo_xy) 
            dispmap_erp = depthmap_utils.depth2disparity(depthmap_erp).astype(np.float32)

                        
            self.depthmap_persp_list.append(depthmap_persp)
            self.depthmap_erp_list.append(depthmap_erp) 
            self.dispmap_erp_list.append(dispmap_erp)

        
        depthmap_utils.depth_visual_save(erp_pred, f'{self.depth_align_dir}/erp_pred.png')
        depthmap_utils.depth_visual_save(self.erp_mask, f'{self.depth_align_dir}/erp_mask_pred.png')

        depthmap_utils.depth_ico_visual_save(self.dispmap_persp_list, f"{self.depth_align_dir}/dispmap_persp.png") 
        depthmap_utils.depth_ico_visual_save(self.depthmap_persp_list, f"{self.depth_align_dir}/depthmap_persp.png") 
        depthmap_utils.depth_ico_visual_save(self.depthmap_erp_list, f"{self.depth_align_dir}/depthmap.png") 
        depthmap_utils.depth_ico_visual_save(self.dispmap_erp_list, f"{self.depth_align_dir}/dispmap.png")
 
 
    @torch.no_grad()
    def depth_alignment(self):
        # from utils import depthmap_align
        depthmap_aligner = DepthmapAlign(self.opt, self.depth_align_dir, self.subimg_rgb_list, debug=True)
             
        subimage_available_list = list(range(len(self.dispmap_erp_list)))
        self.dispmap_aligned_list, coeffs_scale, coeffs_offset, self.subimg_cam_list = \
            depthmap_aligner.align_multi_res(self.pano_rgb, self.dispmap_erp_list, self.opt.subimage_padding_size, subimage_available_list)
         
        blend_it = blending.BlendIt(self.opt.subimage_padding_size, len(subimage_available_list), self.opt.blending_method)
        blend_it.fidelity_weight = 0.1
        
        pano_rgb_height = self.pano_rgb.shape[0]
        blend_it.tangent_images_coordinates(pano_rgb_height, self.dispmap_aligned_list[0].shape)
        blend_it.erp_blendweights(self.subimg_cam_list, pano_rgb_height, self.dispmap_aligned_list[0].shape)
        blend_it.compute_linear_system_matrices(pano_rgb_height, pano_rgb_height * 2, blend_it.frustum_blendweights)

        erp_dispmap_blend = blend_it.blend(self.dispmap_aligned_list, pano_rgb_height)
        blending_method = 'frustum' if self.opt.blending_method == 'all' else self.opt.blending_method
        erp_dispmap_blend_save = erp_dispmap_blend[blending_method]  

        shutil.rmtree(self.depth_align_dir)

        if self.persp_monodepth != 'zoedepth': 
            erp_depthmap_blend_save = np.full(erp_dispmap_blend_save.shape, 0, np.float64)
            bias = erp_dispmap_blend_save[self.erp_mask].max() - erp_dispmap_blend_save[self.erp_mask].min()

            erp_depthmap_blend_save[self.erp_mask] = erp_dispmap_blend_save[self.erp_mask].max() - erp_dispmap_blend_save[self.erp_mask] + bias / 4
            erp_depthmap_blend_save[self.erp_mask] = erp_depthmap_blend_save[self.erp_mask] / erp_depthmap_blend_save[self.erp_mask].max()
            max_no_zeros = erp_depthmap_blend_save[self.erp_mask].max()
            erp_depthmap_blend_save[~self.erp_mask] = max_no_zeros * 1            
            
            return erp_depthmap_blend_save
        else:

            erp_dispmap_blend_save =  (erp_dispmap_blend_save - erp_dispmap_blend_save.min()) / (erp_dispmap_blend_save.max() - erp_dispmap_blend_save.min()) 
            np.save(os.path.join(self.save_dir, 'erp_dispmap_blend_save.npy'), erp_dispmap_blend_save)
            print("============= Pass ============")
            return erp_dispmap_blend_save       
        
        # pointcloud_utils.depthmap2pointcloud_erp(erp_dispmap_blend_save, self.pano_rgb, f"{self.log_dir}/pointcloud_rgb.ply")
        # depthmap_utils.depth_visual_save(erp_dispmap_blend_save, f"{self.log_dir}/pano_depth.png")
  
  
  
  
  
  