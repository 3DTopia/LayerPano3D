from utility import depth_stitch, serialization
from utility import depthmap_utils
from utility import serialization
from utility import image_io
from utility import fs_utility
from utility import subimage

import numpy as np
import projection_icosahedron as proj_ico

import pathlib
import os

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def create_alignment_data(erp_depthmap, subimage_size=500,
                          padding_size=0.3,
                          coeff_scale_list=None, coeff_offset_list=None, grid_size=[5, 5]):
    """Create the depth alignment test data.

    Project the ERP depth map to 20 sub-images, with the scale and offset.
  
    """
    # 0) check the scale and offset coefficients
    if coeff_scale_list is None:
        rng = np.random.default_rng(seed=0)
        coeff_scale_list = rng.random((grid_size, 20))

    if coeff_offset_list is None:
        rng = np.random.default_rng(seed=1)
        coeff_offset_list = rng.random((grid_size, 20))

    # 1) erp to 20 images
    subimage_list, _, _ = proj_ico.erp2ico_image(erp_depthmap, subimage_size, padding_size, full_face_image=True)

    # 2) scale and offset the subimage disparity map
    subimage_list_new = []
    for idx in range(len(subimage_list)):
        temp = depth_stitch.depthmap_deform(subimage_list[idx], coeff_scale_list[idx], coeff_offset_list[idx])
        subimage_list_new.append(temp)

    return subimage_list_new, coeff_scale_list, coeff_offset_list


def data_visualizing(mock_data_root_dir, frame_number):
    """
    """
    # 0) visualize the depth map (*.pfm)
    filenameConv = fs_utility.FileNameConvention()
    filenameConv.set_filename_basename("img0")
    filenameConv.set_filepath_folder(mock_data_root_dir)

    depthmap_list = {}
    for idx in range(frame_number):
        depthmap_filename = pathlib.Path(filenameConv.subimage_depthmap_filename_expression.format(idx))
        dispmap, _ = depthmap_utils.read_pfm(str(depthmap_filename))
        dispmap_vis_filepath = str(depthmap_filename) + ".jpg"
        depthmap_utils.depth_visual_save(dispmap, str(dispmap_vis_filepath))
        log.debug("visualize disparity map {}".format(str(depthmap_filename)))

        depthmap_list[idx] = dispmap

    image_height = depthmap_list[0].shape[0]
    image_width = depthmap_list[0].shape[1]

    # 1) visualize the pixel corresponding relationship (*.json)
    for src_idx in range(frame_number):
        for tar_idx in range(frame_number):
            if src_idx == tar_idx:
                continue
            filename_src2tar = filenameConv.subimage_pixelcorr_filename_expression.format(src_idx, tar_idx)
            pixels_corresponding = serialization.pixel_corresponding_load(filename_src2tar)
            # print("load the pixels corresponding file: {}".format(filename_src2tar))

            src_image_filepath = pixels_corresponding["src_image_filename"]
            tar_image_filepath = pixels_corresponding["tar_image_filename"]

            # load subimage
            if os.path.isabs(src_image_filepath) and os.path.isabs(tar_image_filepath):
                # src_image_data = image_io.image_read(src_image_filepath + ".jpg")
                # tar_image_data = image_io.image_read(tar_image_filepath+ ".jpg")
                image1_output_path = src_image_filepath + "_{}_{}.jpg".format(src_idx, tar_idx)
                image2_output_path = tar_image_filepath + "_{}_{}.jpg".format(tar_idx, src_idx)
            else:
                # src_image_data = image_io.image_read(mock_data_root_dir + src_image_filepath+ ".jpg")
                # tar_image_data = image_io.image_read(mock_data_root_dir + tar_image_filepath+ ".jpg")
                image1_output_path = mock_data_root_dir + src_image_filepath + "_{}_{}.jpg".format(src_idx, tar_idx)
                image2_output_path = mock_data_root_dir + tar_image_filepath + "_{}_{}.jpg".format(tar_idx, src_idx)

            src_image_data = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
            tar_image_data = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
            # plot the pixels points
            pixel_corresponding_array = pixels_corresponding["pixel_corresponding"]
            pixel_corresponding_number = pixels_corresponding["pixel_corresponding_number"]

            if pixel_corresponding_array is None or pixel_corresponding_number <= 0:
                continue

            src_image_data_image_np, tar_image_data_image_np, _ = subimage.draw_corresponding(src_image_data, tar_image_data, pixel_corresponding_array)

            image_io.image_save(src_image_data_image_np, image1_output_path)
            image_io.image_save(tar_image_data_image_np, image2_output_path)
            # , image1_output_path, image2_output_path

    # 2) visualize the scale and offset coefficients (*.json)
