import cv2
from utility import subimage
from utility import depthmap_utils
from utility import serialization
from utility import image_io

from skimage.transform import pyramid_gaussian
import numpy as np

import os
import pickle

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False

try:
    from instaOmniDepth import depthmapAlign
    log.info("depthmapAlign python module installed!")
except ModuleNotFoundError:
    log.error("depthmapAlign python module do not install, please build and install it reference the readme.")


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

    def __init__(self):
        # sub-image number
        self.depthmap_number = -1
        self.output_dir = ""  # output alignment coefficient
        self.depthmap_aligned = None

        # output debug information filepath exp
        self.debug = False
        # if output path is None do not output the data to files
        self.subimage_pixelcorr_filepath_expression = None
        self.subimage_depthmap_aligning_filepath_expression = None  # save the normalized depth map for each resolution
        self.subimage_warpedimage_filepath_expression = None
        self.subimage_warpeddepth_filename_expression = None
        self.subimage_alignment_intermedia_filepath_expression = None   # pickle file
        self.subimages_rgb = None  # the original 20 rgb subimages, if is not warp depth map

        # align grid
        self.align_coeff_grid_width = 5           # the width of the initial grid
        self.align_coeff_grid_height = 8          # the height of the initial grid
        self.align_coeff_grid_width_finest = 10   # the grid width of the finest grid
        self.align_coeff_grid_height_finest = 16  # the grid height of the finest grid.
        self.align_coeff_initial_scale_list = []
        self.align_coeff_initial_offset_list = []
        self.depthmap_original_ico_index = []      # the subimage's depth map ico face index

        # ceres options
        self.ceres_thread_number = 12
        self.ceres_max_num_iterations = 25
        self.ceres_max_linear_solver_iterations = 10
        self.ceres_min_linear_solver_iterations = -1

        # align parameter
        self.align_method = "group"
        self.weight_project = 1.0
        self.weight_smooth = 0.1
        self.weight_scale = 0.0
        self.depthmap_norm_mothod = "midas"
        self.coeff_fixed_face_index = 7

        # multi-resolution parameters
        self.multi_res_grid = False
        self.pyramid_layer_number = 1
        self.pyramid_downscale = 2

        # pixel correpsonding down-sample parameter
        self.downsample_pixelcorr_ratio = 0.4

        # initial the align process (depthmapAlign) run time
        depthmapAlign.init(self.align_method)
        # clear the alignment run time, call depthmapAlign.shutdown when the interpreter exits
        import atexit
        atexit.register(depthmapAlign.shutdown)

        # the global configuration
        self.opt = None

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

        print("Re-projection cost is {}, per pixel is {}".format(diff_sum, diff_sum / pixel_numb))

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
                    subimage.erp_ico_proj(erp_rgb_image_data, padding_size, tangent_image_width, self.downsample_pixelcorr_ratio, self.opt)

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
