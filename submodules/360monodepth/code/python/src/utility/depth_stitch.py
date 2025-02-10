import cam_models

from scipy import ndimage
import numpy as np


from logger import Logger
log = Logger(__name__)
log.logger.propagate = False


def depthmap_deform(depthmap, scale_array, offset_array):
    """    Deform the depth map with scale and offset.


    :param depthmap: The original depth map [height, width]
    :type depthmap: numpy 
    :param scale_array: The offset array[height, width]
    :type scale_array: numpy
    :param offset_array: The offset array, [height, width]
    :type offset_array: numpy
    :return: deformed depth map
    :rtype: numpy
    """    
    if scale_array.shape != offset_array.shape:
        log.error("The offset size is not equal scale's shape.")

    image_height = depthmap.shape[0]
    image_width = depthmap.shape[1]
    
    # 0) get scale and offset for each pixel
    corr_yv, corr_xv = np.mgrid[0:image_height, 0:image_width]
    corr_yv_grid = corr_yv / (image_height - 1) * (scale_array.shape[0] - 1)
    corr_xv_grid = corr_xv / (image_width - 1) * (scale_array.shape[1] - 1)

    scale_pixelwise = ndimage.map_coordinates(scale_array, [corr_yv_grid, corr_xv_grid], order=1, mode='constant', cval=0.0)
    offset_pixelwise = ndimage.map_coordinates(offset_array, [corr_yv_grid, corr_xv_grid], order=1, mode='constant', cval=0.0)

    # 1) get new depth map
    dephtmap_deformed = depthmap * scale_pixelwise + offset_pixelwise
    return dephtmap_deformed


def stitch_depth_subimage(depth_data_list, image_param_list, fisheye_model):
    """Stitch perspective images to fisheye image.

    :param depth_data_list: The perspective images data.
    :type depth_data_list: list
    :param image_param_list: The perspective images parameters.
    :type image_param_list: list
    :param fisheye_model: the fisheye image parameters. It's Brown data fromat.
    :type fisheye_model: dict
    :return: the stitched fisheye image.
    :rtype: numpy
    """
    # get the fisheye image size
    fisheye_image_height = fisheye_model["intrinsics"]["image_size"][0]
    fisheye_image_width = fisheye_model["intrinsics"]["image_size"][1]

    fisheye_depth = np.zeros((fisheye_image_height, fisheye_image_width), float)
    fisheye_image_weight = np.zeros((fisheye_image_height, fisheye_image_width), float)

    # project the pinhole image to 3D
    x_list = np.linspace(0, fisheye_image_width, fisheye_image_width, endpoint=False)
    y_list = np.linspace(0, fisheye_image_height, fisheye_image_height, endpoint=False)
    grid_x, grid_y = np.meshgrid(x_list, y_list)
    fisheye_2d_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    fisheye_3d_points = cam_models.cam2world(fisheye_2d_points, fisheye_model)

    for index in range(0, len(depth_data_list)):
        image_param = image_param_list[index]
        depth_data = depth_data_list[index]

        pinhole_image_height = depth_data.shape[0]
        pinhole_image_width = depth_data.shape[1]

        # # Transform pixel coords to world coords (spherical 3D points)
        pinhole_3d_points_original = image_param['rotation'] @ fisheye_3d_points.T  # + image_param['translation'][:, np.newaxis]
        pinhole_3d_points = (np.divide(pinhole_3d_points_original, pinhole_3d_points_original[2, :]))
        pinhole_2d_points = (image_param["intrinsics"]["matrix"] @ pinhole_3d_points)[:2, :]

        # just use  the available 3D point, in +z and have corresponding 2D image points
        available_pixels_list = np.logical_and.reduce((
            pinhole_3d_points_original[2, :] > 0,
            pinhole_2d_points[0, :] >= 0, pinhole_2d_points[0, :] < pinhole_image_width,
            pinhole_2d_points[1, :] >= 0, pinhole_2d_points[1, :] < pinhole_image_height))

        available_pixels_list_mat = available_pixels_list.reshape((fisheye_image_height, fisheye_image_width))
        fisheye_depth[:, :][available_pixels_list_mat] += ndimage.map_coordinates(
            depth_data[:, :],
            [pinhole_2d_points[1, :][available_pixels_list], pinhole_2d_points[0, :][available_pixels_list]],
            order=1, mode='constant', cval=0.0)

        fisheye_image_weight[available_pixels_list_mat] += 1

    # get pixels mean
    available_weight_list = fisheye_image_weight != 0
    fisheye_depth[:, :][available_weight_list] = np.divide(fisheye_depth[:, :][available_weight_list], fisheye_image_weight[available_weight_list])

    return fisheye_depth


def find_corresponding(src_image, src_param, tar_image, tar_param, fisheye_model):
    """Get two images' pixels corresponding relationship.

    Input image size is [height, width, 2].

    :param src_image: source image data array
    :type src_image: numpy
    :param src_param: source image's intrinsic and extrinsic parameters
    :type src_param: dict
    :param tar_image:  target image data array
    :type tar_image: numpy
    :param tar_param: target image's intrinsic and extrinsic parameters
    :type tar_param: dict
    :param fisheye_model: fisheye model.
    :type fisheye_model: dict
    :return: dict record the pixels relationship,[points_number, 4], [src_y, src_x, tar_y, tar_x]
    :rtype: dict
    """
    #
    pixel_matching = []  # recording the pixel corresponding [4, pixels_number]
    src_image_height = src_image.shape[0]
    src_image_width = src_image.shape[1]
    tar_image_height = tar_image.shape[0]
    tar_image_width = tar_image.shape[1]
    if src_image_height != tar_image_height or tar_image_width != src_image_width:
        log.error("the source image size is not same as the target image size.")

    # 1) project source image pinhole image to fisheye image
    src_image_u_list = np.linspace(0, src_image_width, src_image_width, endpoint=False)  # x
    src_image_v_list = np.linspace(0, src_image_height, src_image_height, endpoint=False)  # y
    src_image_grid_u, src_image_grid_v = np.meshgrid(src_image_u_list, src_image_v_list)
    src_image_grid_z = np.ones(src_image_grid_u.shape, float)
    src_image_2d_points = np.stack((src_image_grid_u.ravel(), src_image_grid_v.ravel(), src_image_grid_z.ravel()), axis=1)

    # project the pinhole image to world coords (spherical 3D points)
    src_image_3d_points = np.linalg.inv(src_param['intrinsics']['matrix']) @ src_image_2d_points.T
    src_image_3d_points_world = np.linalg.inv(src_param['rotation']) @ (src_image_3d_points - src_param['translation'][:, np.newaxis])
    fisheye_2d_points = cam_models.world2cam_slow(src_image_3d_points_world, fisheye_model).T

    # 2) from fisheye image to target image
    fisheye_3d_points = cam_models.cam2world(fisheye_2d_points, fisheye_model)
    # pinhole_3d_points = (np.divide(fisheye_3d_points, fisheye_3d_points[2, :]))
    tar_image_3d_points = tar_param['rotation'] @ fisheye_3d_points.T + tar_param['translation'][:, np.newaxis]

    # projection to pin-hole image
    tar_image_3d_points = (np.divide(tar_image_3d_points, tar_image_3d_points[2, :]))
    tar_image_2d_points = (tar_param["intrinsics"]["matrix"] @ tar_image_3d_points)[:2, :]

    # get the available pixels index
    available_pixels_list = np.logical_and.reduce((
        tar_image_2d_points[0, :] >= 0, tar_image_2d_points[0, :] <= tar_image_width - 1,
        tar_image_2d_points[1, :] >= 0, tar_image_2d_points[1, :] <= tar_image_height - 1))

    # 3) check the similarity of the corresponding pixels
    tar_image_2d_points = tar_image_2d_points.T
    src_image_2d_points_available = src_image_2d_points[available_pixels_list][:, :2]
    tar_image_2d_points_available = tar_image_2d_points[available_pixels_list]

    # x-y to y-x
    src_image_2d_points_available = src_image_2d_points_available[:, [1, 0]]
    tar_image_2d_points_available = tar_image_2d_points_available[:, [1, 0]]

    if src_image_2d_points_available.shape[0] == 0:
        log.debug("the do not have overlap between two images.")
    else:
        src_image_avail_pixel_data = src_image[src_image_2d_points_available.astype(int)]
        tar_image_avail_pixel_data = tar_image[tar_image_2d_points_available.astype(int)]
        rms = np.sqrt(np.mean((src_image_avail_pixel_data - tar_image_avail_pixel_data) ** 2))
        log.debug("The corresponding pixel rms is {}".format(rms))

    # 4) save to numpy array
    # src_image_2d_points_available_np = np.array(src_image_2d_points_available)
    # tar_image_2d_points_available_np = np.array(tar_image_2d_points_available)
    pixel_matching = np.hstack((src_image_2d_points_available, tar_image_2d_points_available))
    return pixel_matching
