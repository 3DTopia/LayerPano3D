from MiDaS.monodepth_net import MonoDepthNet
from MiDaS.run import *
import MiDaS.MiDaS_utils as MiDaS_utils

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('TkAgg')

import sys
import os
import json


from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

def world2cam_slow(point3D, cam_model):
    """
    code is from: https://github.com/matsuren/ocamcalib_undistort

    Our coordinate system is 
    The original code's 3D coordinate system, x:right direction, y:down direction, z:front direction
    # is [y,x,z]

    return is [x, y]
    """
    """ world2cam(point3D) projects a 3D point on to the image.
    If points are projected on the outside of the fov, return (-1,-1).
    Also, return (-1, -1), if point (x, y, z) = (0, 0, 0).
    The coordinate is different than that of the original OcamCalib.
    point3D coord: x:right direction, y:down direction, z:front direction
    point2D coord: x:row direction, y:col direction (OpenCV image coordinate).

    Parameters
    ----------
    point3D : numpy array or list([x, y, z]), in OpenCV coordinate system.
        array of points in camera coordinate (3xN)

    Returns
    -------
    point2D : numpy array,  in OpenCV coordinate system.
        array of points in image (2xN)

    Examples
    --------
    >>> ocam = OcamCamera('./calib_results_0.txt')
    >>> ocam.world2cam([1,1,2.0]).tolist() # project a point on image
    [[1004.8294677734375], [1001.1594848632812]]
    >>> tmp = ocam.world2cam(np.random.rand(3, 10)) # project multiple points without error
    >>> ocam.world2cam([0,0,2.0]).tolist() # return optical center
    [[798.1757202148438], [794.3086547851562]]
    >>> ocam.world2cam([0,0,0]).tolist()
    [[-1.0], [-1.0]]
    """
    # in case of point3D = list([x,y,z])
    if isinstance(point3D, list):
        point3D = np.array(point3D)
    if point3D.ndim == 1:
        point3D = point3D[:, np.newaxis]
    assert point3D.shape[0] == 3

    # transform the OpenCV coordinate system to OCamCalib's 
    x_ocamcalib = point3D[1]
    y_ocamcalib = point3D[0]
    z_ocamcalib = -point3D[2]

    # 0) transfrom the camera parameter
    _yc = cam_model['intrinsics']['distortion_center'][1]
    _xc = cam_model['intrinsics']['distortion_center'][0]
    _invpol = np.array(cam_model['intrinsics']['poly'])[::-1]
    _affine = np.array(cam_model['intrinsics']['stretch_matrix'])
    _fov = 185

    # return value
    point2D_ocamcalib = np.zeros((2, x_ocamcalib.shape[0]), dtype=np.float32)

    norm = np.sqrt(x_ocamcalib * x_ocamcalib + y_ocamcalib * y_ocamcalib)
    valid_flag = (norm != 0)

    # optical center
    point2D_ocamcalib[0][~valid_flag] = _xc
    point2D_ocamcalib[1][~valid_flag] = _yc
    # point = (0, 0, 0)
    # zero_flag = (point3D == 0).all(axis=0)
    zero_flag = np.logical_and.reduce((x_ocamcalib == 0, y_ocamcalib == 0 , z_ocamcalib == 0))

    point2D_ocamcalib[0][zero_flag] = -1
    point2D_ocamcalib[1][zero_flag] = -1

    # else
    theta = np.arctan(z_ocamcalib[valid_flag] / norm[valid_flag])
    invnorm = 1 / norm[valid_flag]
    rho = 0
    tmp_theta = None
    #     rho = np.array([element * theta ** i for (i, element) in enumerate(self._invpol)]).sum(axis=0) is slow
    for (i, element) in enumerate(_invpol):
        if i == 0:
            rho = np.full_like(theta, element)
            tmp_theta = theta.copy()
        else:
            rho += element * tmp_theta
            tmp_theta *= theta

    u = x_ocamcalib[valid_flag] * invnorm * rho
    v = y_ocamcalib[valid_flag] * invnorm * rho
    point2D_valid_0 = v * _affine[2] + u + _yc
    point2D_valid_1 = v * _affine[0] + u * _affine[1] + _xc

    if _fov < 360:
        # finally deal with points are outside of fov
        thresh_theta = np.deg2rad(_fov / 2) - np.pi / 2
        # set flag when  or point3D == (0, 0, 0)
        outside_flag = theta > thresh_theta
        point2D_valid_0[outside_flag] = -1
        point2D_valid_1[outside_flag] = -1

    point2D_ocamcalib[0][valid_flag] = point2D_valid_0
    point2D_ocamcalib[1][valid_flag] = point2D_valid_1

    # change the 2D points from OCamCalib to OpenCV coordinate system
    point2D_opencv = point2D_ocamcalib[[1,0],:]
    return point2D_opencv


def world2cam(matrix_3d, cam_model):
    """
    TODO check the coordinate system. Make it same as world2cam_slow

    World to pixel coordinates function from Scaramuzza
    :param matrix_3d: points in world coords
    :param cam_model: camera parameters
    :return: projected points in pixel coordinates
    """
    # R = np.array(cam_model['rotation']).reshape(3, 3)
    # t = np.array(cam_model['translation'])
    invpol = np.array(cam_model['intrinsics']['fast_poly'])[::-1]
    #   distorsion_c s inverted because the image is now rotated (3840-height, 2880-width)
    distorsion_c = np.array(cam_model['intrinsics']['distortion_center'])[::-1]
    A = np.array(cam_model['intrinsics']['stretch_matrix']).reshape(2, 2)
    length_invpol = len(invpol)

    norm = np.linalg.norm(matrix_3d[:, :-1], axis=1)
    norm[np.where(norm == 0)] = sys.float_info.epsilon
    theta = np.arctan(np.divide(matrix_3d[:, -1], norm))

    rho = np.ones(norm.shape)*invpol[0]
    t_i = np.ones(norm.shape)

    for i in range(1, length_invpol):
        t_i *= theta
        rho += np.multiply(t_i, invpol[i])

    u = matrix_3d[:, 1]/norm*rho
    v = matrix_3d[:, 0]/norm*rho

    point2D = np.dstack((v, u)).reshape(-1, 2)
    point2D = point2D @ A.T + distorsion_c

    return point2D

    # invpol = np.array([730.949123, 315.876984, -177.960849, -352.468231, -678.144608, -615.917273,
    # -262.086205, -42.961956 ])
    # invpol = np.array([8.8382e3, -6.6872e4, 2.2249e5, -4.2715e5, 5.2136e5, -4.1954e5, 2.2254e5, -7.5488e4, 1.4624e4])[::-1]


def cam2world(matrix_2d, cam_model):
    """
    Project the fisheye image to spherical coordinate 3D point.
    Input and output data coordinate is OpenCV coordinate system.
    Fisheye camera parameters are OcamCalib coordinate system.

    cite from OCamCalib official webpage:
    `
        Back-projects a pixel point m onto the unit sphere M.
        M=[X;Y;Z] is a 3xN matrix with which contains the coordinates of the vectors emanating from the single-effective-viewpoint to the unit sphere, therefore, X^2 + Y^2 + Z^2 = 1.
    `

    More please refer https://sites.google.com/site/scarabotix/ocamcalib-toolbox

    :param matrix_2d: pixel coordinates in OpenCV coordinate system [pointnumber ,2 ] , [:,0] is x , [:,1] is y
    :type matrix_2d: numpy
    :param cam_model: camera parameters, in OCamCalib coordinate system (convention).
    :type cam_model dict
    :return: 3D points in OpenCV coordinate system [x,y,z]
    :rtype: numpy
    """
    # OpenCV to OCamClib coordinate system
    matrix_2d_ocamclib = matrix_2d[:,[1,0]]

    # The following is OCamClib coordinate system
    # all parameter store in OCamClib coordinate system, because it computed by OCamCablib tool-box
    pol = np.array(cam_model['intrinsics']['mapping_coefficients'])
    distorsion_c = np.array(cam_model['intrinsics']['distortion_center'])
    # A = np.array(cam_model['intrinsics']['stretch_matrix']).reshape(2, 2)
    c = cam_model['intrinsics']['stretch_matrix'][0]
    d = cam_model['intrinsics']['stretch_matrix'][1]
    e = cam_model['intrinsics']['stretch_matrix'][2]

    invdet = 1 / (c - d * e)    # 1/det(A), where A = [c,d;e,1] as in the Matlab file
    # point2D = matrix_2d @ A.I - distorsion_c
    u_ocam = invdet * ((matrix_2d_ocamclib[:, 0] - distorsion_c[0]) - d * (matrix_2d_ocamclib[:, 1] - distorsion_c[1]))
    v_ocam = invdet * (-e * (matrix_2d_ocamclib[:, 0] - distorsion_c[0]) + c * (matrix_2d_ocamclib[:, 1] - distorsion_c[1]))
    # point2D = matrix_2d @ A.I - distorsion_c
    # point2D = matrix_2d @ np.linalg.inv(A) - distorsion_c
    # point2D = matrix_2d @ np.linalg.inv(A) - distorsion_c
    # point2D = np.empty_like(matrix_2d)
    # point2D[:, 0] = xp
    # point2D[:, 1] = yp

    # norm = np.linalg.norm(matrix_2d, axis=1)
    norm = np.sqrt(u_ocam * u_ocam + v_ocam * v_ocam)
    z = np.ones(norm.shape)*pol[0]
    r_i = np.ones(norm.shape)

    for i in range(1, len(pol)):
        r_i *= norm
        z += r_i * pol[i]

    # change the spherical 3D points from OCamCalib to OpenCV coordinate system
    u_opencv = v_ocam
    v_opencv = u_ocam
    z_opencv = -z
    points3D = np.stack((u_opencv ,v_opencv , z_opencv)).T

    # normalize to unit norm
    points3D /= np.linalg.norm(points3D, axis=1)[:, np.newaxis]
    return points3D


def create_perspective_undistortion_LUT(img_shape, cam_model, sf=4, patch=None):
    """
    This function is used in case we want to undistort the whole fisheye image or just a patch without taking
    perspective subimages

    :param img_shape: in matrix form (rows, cols)
    :param cam_model: camera parameters
    :param sf: scale factor. From Scaramuzza: it works as a zoom factor?
    :param patch: ROI to undistory. If None it undistorts the whole image
    :return: undistorted image
    """
    width = img_shape[1]  # New width
    height = img_shape[0]  # New height
    Nyc = height/2.0
    Nxc = width/2.0
    Nz  = width/sf

    if patch is not None:
        aux_x = patch[..., 0]
        aux_y = patch[..., 1]
    else:
        aux_y, aux_x = np.mgrid[0:height, 0:width]

    aux_y = aux_y - Nyc
    aux_x = aux_x - Nxc
    aux_z = -np.ones(aux_y.shape) * Nz

    matrix_3d = np.dstack((aux_x.flatten(), aux_y.flatten(), aux_z.flatten()))
    matrix_3d = matrix_3d.reshape(-1, 3)
    matrix_2d = world2cam(matrix_3d, cam_model)

    if patch is not None:
        matrix_2d = matrix_2d.reshape((patch.shape[1], patch.shape[0], 2))
    else:
        matrix_2d = matrix_2d.reshape((height, width, 2))
    return matrix_2d[..., 0], matrix_2d[..., 1]


def point3d2obj(points_3d, obj_path):
    """
    show ocamcalib's 3d points.
    """
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x =fisheye_3d_points[:,0]
    # y =fisheye_3d_points[:,1]
    # z =fisheye_3d_points[:,2]
    # ax.scatter(x, y, z, c='r', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()
    with open(obj_path, "w") as f:
        for index in range(0,points_3d.shape[0]):
            f.write("v {} {} {}\n".format(points_3d[index, 0],points_3d[index, 1],points_3d[index,2]))

    
def stitch_rgb_image(image_data_list, image_param_list, fisheye_model, subimage_fov=60, fisheye_fov=180):
    """Stitch perspective images to fisheye image.

    :param image_data_list: The perspective images data.
    :type image_data_list: list
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
    channel_number = image_data_list[0].shape[2]

    # project the pinhole image to world coords (spherical 3D points)
    x_list = np.linspace(0, fisheye_image_width, fisheye_image_width, endpoint=False)  
    y_list = np.linspace(0, fisheye_image_height, fisheye_image_height, endpoint=False)  
    grid_x, grid_y = np.meshgrid(x_list, y_list)
    fisheye_2d_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
    fisheye_3d_points = cam2world(fisheye_2d_points, fisheye_model) 
    # point3d2obj(fisheye_3d_points, "D:/1.obj")

    fisheye_image = np.zeros((fisheye_image_height, fisheye_image_width, channel_number), float)
    fisheye_image_weight = np.zeros((fisheye_image_height, fisheye_image_width), float)
    for index in range(0, len(image_data_list)):
        fisheye_image_weight_subimg = np.zeros((fisheye_image_height, fisheye_image_width), float)
        fisheye_image_subimage = np.zeros((fisheye_image_height, fisheye_image_width, channel_number), float)

        image_param = image_param_list[index]
        image_data = image_data_list[index]

        pinhole_image_height = image_data.shape[0]
        pinhole_image_width = image_data.shape[1]

        pinhole_3d_points = image_param['rotation'] @ fisheye_3d_points.T + image_param['translation'][:, np.newaxis]

        # check the pixel in the available hfov and vfov range of fisheye image
        hfov = subimage_fov
        vfov = subimage_fov
        radius = np.linalg.norm(pinhole_3d_points.T, axis=1)
        points_azimuth = np.degrees(np.arctan2(pinhole_3d_points[0, :], pinhole_3d_points[2, :]))
        points_altitude = np.degrees(np.arcsin(np.divide(-pinhole_3d_points[1, :], radius)))

        points_azimuth_inhfov = np.logical_and(points_azimuth > - 0.5 * hfov, points_azimuth < 0.5 * hfov)
        points_altitude_invfov = np.logical_and(points_altitude > - 0.5 * vfov, points_altitude < 0.5 * vfov)
        available_pixels_list_fov = np.logical_and(points_azimuth_inhfov, points_altitude_invfov)
        available_pixels_list_fov_mat = available_pixels_list_fov.reshape(fisheye_image_height, fisheye_image_width)

        fisheye_2d_points_subimage = fisheye_2d_points[available_pixels_list_fov].astype(int)

        # projection to pin-hole image
        pinhole_3d_points = (np.divide(pinhole_3d_points, pinhole_3d_points[2, :]))
        pinhole_2d_points = (image_param["intrinsics"]["matrix"] @ pinhole_3d_points)[:2, :]
        pinhole_2d_points = pinhole_2d_points[:, available_pixels_list_fov]

        # just use the 3D point in the available pinhole image range
        available_pixels_list = np.logical_and.reduce((
            pinhole_3d_points[2, :][available_pixels_list_fov] > 0,
            pinhole_2d_points[0, :] >= 0, pinhole_2d_points[0, :] <= pinhole_image_width - 1,
            pinhole_2d_points[1, :] >= 0, pinhole_2d_points[1, :] <= pinhole_image_height - 1))

        for channel in range(0, channel_number):
            fisheye_image_subimage[fisheye_2d_points_subimage[:, 1][available_pixels_list], fisheye_2d_points_subimage[:, 0][available_pixels_list], channel] = \
                ndimage.map_coordinates(image_data[:, :, channel],
                                        [pinhole_2d_points[1, :][available_pixels_list], pinhole_2d_points[0, :][available_pixels_list]],
                                        order=1, mode='constant', cval=255.0)

        # compute blend weight
        available_pixels_weight = np.ones(available_pixels_list.shape,  float)
        available_pixels_weight[~available_pixels_list] = 0
        fisheye_image_weight_subimg[available_pixels_list_fov_mat] = available_pixels_weight

        fisheye_image_weight += fisheye_image_weight_subimg
        fisheye_image += fisheye_image_subimage

    # get pixels mean
    available_weight_list = fisheye_image_weight != 0
    for channel in range(0, channel_number):
        fisheye_image[:, :, channel][available_weight_list] = np.divide(fisheye_image[:, :, channel][available_weight_list], fisheye_image_weight[available_weight_list])

    return fisheye_image.astype(np.uint8)


def sample_rgb_image(img, model, fov=[60, 60], canvas_size=[400, 400], sample_grid_size=[3, 3]):
    """ Sample perspective images from fish-eye image.

    :param img: fisheye rgb image data
    :type img: numpy
    :param model: Scaramuzza camera model
    :type model: dict
    :param fov: fov for the perspective views, [h_fov, v_fov]
    :type fov: list
    :param canvas_size: the perspective images size, [image_height, image_width]
    :type canvas_size: list
    :param sample_grid_size: the sample grid size [horizontal grid, vertical grid].
    :type sample_grid_size: list
    :return: perspective images and camera parameters.
    """
    hfov_fisheye = 180  # TODO load from camera_model dict
    vfov_fisheye = 180
    hfov_pinhole = fov[0]
    vfov_pinhole = fov[1]
    horizontal_size = sample_grid_size[0]
    vertical_size = sample_grid_size[1]
    image_height_pinhole = canvas_size[0]
    image_width_pinhole = canvas_size[1]

    # 0) generate the perspective camera parameters
    # 0-0) get the camera orientation
    xyz_rotation_array = generate_camera_orientation(hfov_fisheye, vfov_fisheye, hfov_pinhole, vfov_pinhole, horizontal_size, vertical_size, 20)

    # 0-1) get the camera intrinsic and extrinsic parameters
    sub_image_param_list = get_perspective_camera_parameters(
        hfov_pinhole, vfov_pinhole, image_width_pinhole, image_height_pinhole,  xyz_rotation_array.T)

    # 1) generate the perspective images
    # TODO perspective image size is base on fov ane original fish-eye image size
    canvas_yy, canvas_xx = np.mgrid[0:image_height_pinhole, 0:image_width_pinhole]
    canvas_2d = np.squeeze(np.dstack((canvas_xx.flatten(), canvas_yy.flatten())))
    # canvas_2d each column is 3D point [x,y,1]
    canvas_2d = np.hstack((canvas_2d, np.ones((image_height_pinhole * image_width_pinhole, 1))))
    sub_images = []
    channel_number = img.shape[2]

    for index in range(0, len(sub_image_param_list)):
        sub_image_param = sub_image_param_list[index]

        # Transform pixel coords to world coords
        pinhole_cs = np.linalg.inv(sub_image_param['intrinsics']['matrix']) @ canvas_2d.T
        world_cs = np.linalg.inv(sub_image_param['rotation']) @ pinhole_cs

        # Fetch RGB from fisheye image
        # NOTE world2cam use fast_poly, world2cam_slow use poly
        # fetch_from = world2cam(world_cs.T, model)
        fetch_from = world2cam_slow(world_cs, model).T
        tangential_img = np.zeros(tuple(canvas_size) + (channel_number,), dtype=float)

        for channel in range(0, channel_number):
            tangential_img[:, :, channel] = ndimage.map_coordinates(img[:, :, channel], [fetch_from[:, 1].reshape(canvas_size), fetch_from[:, 0].reshape(canvas_size)], order=1, mode='constant')

        sub_images.append(tangential_img.astype(np.uint8))

    return sub_images, sub_image_param_list


def sample_img(img, cam_model, fov=53, run_midas=False):
    """
    :param img: fisheye input img
    :param cam_model: Scaramuzza model
    :param fov: fov for the perspective views
    :param run_midas: Set to False in case you already run it once and have the depth maps saved in memory
    :return: perspective subviews. I also call them tangential views
    """

    """
    Thinkin loud...
    Lets wrap up here what I need to do:
    - The goal is to stitch the tangent images (its depth estimation) in the equirectangular domain. How?
        1- Sample the fisheye with each virtual camera getting the tangential images
            1.1) We do this by backprojecting the 2D points to 3D and project these 3D coords back to the fisheye.
            Then, we take the portion of the fisheye correspondent to what the virtual camera sees, i.e. the
            tangential image
        2- We create the empty equirectangular image. We backproject the points to 3D and we find for each 3D point
        its projection into the N virtual cameras. If the 2D projection lies within the camera FOV of ONE of the
        virtuals, then we fetch the tangent image pixel value to its equirectangular equivalent.
        3- What if the equirectangular 3D point lies in more than one virtual camera?? WE NEED BLENDING. Try Poisson.
            3.1- Maybe build an array: (x, y, z, n_proj, [virtual idxs]) where (x,y,z) is the 3D point, n_proj is
            the amount of virtual cameras that point projects to (i.e within its FOV) and [virtual idxs] shows the
            virtual camera indices which contains that point.
            3.2- We should filter that array to only contain the points with n_proj > 1 since those are the only ones 
            for which blending is necessary.
    """
    fisheye_v_fov, fisheye_h_fov = get_fisheyeFOV(img, cam_model)   # Get FOV

    #   Save pinhole camera parameters
    # params_file = open("pinhole_params.json", "w")

    #   TODO: Need to figure out a better way to sample the fisheye domain.
    #   At the moment I just choose a 5x5 virtual array of perspective cameras
    v_angles = np.linspace(-fisheye_v_fov/2, fisheye_v_fov/2, 5)
    h_angles = np.linspace(-fisheye_h_fov/2, fisheye_h_fov/2, 5)[::-1]
    yy, xx = np.meshgrid(v_angles, h_angles, indexing='ij')
    angle_mesh = np.dstack((xx, yy))
    plt.figure()
    plt.imshow(img)
    height, width = img.shape[:-1]

    canvas_size = np.array([384, 384])  # Size for perspective images
    canvas_yy, canvas_xx = np.mgrid[0:canvas_size[0], 0:canvas_size[1]]
    canvas_2d = np.squeeze(np.dstack((canvas_xx.flatten(), canvas_yy.flatten())))
    canvas_2d = np.hstack((canvas_2d, np.ones((canvas_size[0] * canvas_size[1], 1))))

    """
    equirect_3D_points has size (7, 1000x2000)
    Rows [0,1,2] store the 3D points in equirectangular cam coordinates of the equirect image plane. No needed???
    Rows [3,4,5] store the RGB values for the equirect image fetched from the perspective views 
    Rows [6] store the depth values for the equirect image fetched from the estimated depth map of the perspective views
    
    When we finish fetching RGB and depth, the RGB equirect is obtaining by resizing 
    equirect_3D_points[3:6,:] to (3,1000,2000)
    
    and for the depth panorama we need to resize  
    equirect_3D_points[6,:] to (1,1000,2000)
    """

    equirect_size = (3, 1000, 2000)     # Size for equirectangular image
    equirect_3D_points, _ = equirect_cam2world(equirect_size[1:])
    equirect_3D_points_rgb = np.zeros((7, equirect_3D_points.shape[-1]), dtype=float)
    equirect_3D_points_rgb[0, :] = equirect_3D_points[0, :]
    equirect_3D_points_rgb[1, :] = equirect_3D_points[1, :]
    equirect_3D_points_rgb[2, :] = equirect_3D_points[2, :]

    fisheye2equirec = np.zeros((3, equirect_3D_points.shape[-1]), dtype=float)
    #   Lines 200-205 is for converting the whole fisheye to equirectangular
    #   Points at the back of the cylinder are mapped to nan
    nan_boolean = np.bitwise_not(np.isnan(np.sum(equirect_3D_points.T, axis=1)))
    fetch_from = world2cam(equirect_3D_points.T, cam_model).astype(int).T
    fetch_from[0, :] = np.clip(fetch_from[0, :], 0, width - 1)
    fetch_from[1, :] = np.clip(fetch_from[1, :], 0, height - 1)
    fisheye2equirec[:, nan_boolean] = img[fetch_from[1, nan_boolean], fetch_from[0, nan_boolean]].T
    fisheye2equirec = np.moveaxis(fisheye2equirec.ravel().reshape(equirect_size), 0, -1)
    RGB_sub_images = []
    depth_sub_images = []

    if run_midas:
        os.makedirs('../../../../depth_subviews/', exist_ok=True)

    virtual_pinhole_params = []
    #   Iterate through the NxN virtual camera array
    for i in range(0, angle_mesh.shape[0]):
        for j in range(0, angle_mesh.shape[1]):
            idx = i*angle_mesh.shape[1] + j
            angle_pair = angle_mesh[i, j]

            #   Get perspective camera model with angle_pair as parameter for extrinsic params
            pinhole_camera, params_2json = \
                getVirtualCameraMatrix(fov, canvas_size, cam_model, x_angle=angle_pair[1], y_angle=angle_pair[0])
            virtual_pinhole_params.append(params_2json)

            #   Transform pixel coords to world coords
            pinhole_cs = np.linalg.inv(pinhole_camera['intrinsics']['matrix']) @ canvas_2d.T
            world_cs = np.linalg.inv(pinhole_camera['rotation']) @ pinhole_cs

            #   Fetch RGB from fisheye image to assemble perspective subview
            fetch_from = world2cam(world_cs.T, cam_model).astype(int)
            fetch_from[:, 0] = np.clip(fetch_from[:, 0], 0, width-1)
            fetch_from[:, 1] = np.clip(fetch_from[:, 1], 0, height-1)
            virtual2fisheye_idxs = np.dstack((fetch_from[:, 0].reshape(canvas_size), fetch_from[:, 1].reshape(canvas_size)))

            tangential_img = img[virtual2fisheye_idxs[..., 1], virtual2fisheye_idxs[..., 0]]

            if not os.path.isfile('../../../../color_subviews/{:06}_{}_{}.png'.format(idx, int(angle_pair[1]), int(angle_pair[0]))):
                os.makedirs('../../../../color_subviews/', exist_ok=True)
                cv2.imwrite('../../../../color_subviews/frame_{:06}.png'.format(idx), tangential_img)

            if run_midas:
                tangential_depth = run_depth(tangential_img[None], 'MiDaS/model.pt', MonoDepthNet, MiDaS_utils)[0]
                os.makedirs('../../../../depth_subviews/', exist_ok=True)
                np.save('../../../../depth_subviews/frame_{:06}.npy'.format(idx), tangential_depth)
            else:
                tangential_depth = np.load('../../../../depth_subviews/frame_{:06}.npy'.format(idx))

            # tangential_depth = (tangential_depth - np.min(tangential_depth)) / np.ptp(tangential_depth)
            print('frame {}: {}---{}'.format(idx, np.max(tangential_depth), np.min(tangential_depth)))
            RGB_sub_images.append(tangential_img)
            depth_sub_images.append(tangential_depth)

            #   Which equirect 3D points fall in the current perspective subview camera plane?
            equi2tangential_3d = pinhole_camera['intrinsics']['matrix'] @ pinhole_camera['rotation'] @ equirect_3D_points
            equi2tangential = np.divide(equi2tangential_3d, np.tile(equi2tangential_3d[2, :], (3, 1)))[:-1].astype(int)
            inside_fov = (equi2tangential[0, :] >= 0) & (equi2tangential[0, :] < canvas_size[1]) & \
                         (equi2tangential[1, :] >= 0) & (equi2tangential[1, :] < canvas_size[0]) & \
                         (equi2tangential_3d[2, :] >= 0)
            fetch_pixel = equi2tangential[:, inside_fov]

            # Fetch RGB and depth values for the equirect 3D points which fall in the currently evaluated perspective
            # RGB and depth subview respectively
            equirect_3D_points_rgb[3:6, inside_fov] = tangential_img[fetch_pixel[1], fetch_pixel[0]].T
            equirect_3D_points_rgb[6, inside_fov] = tangential_depth[fetch_pixel[1], fetch_pixel[0]].T

    # json.dump(virtual_pinhole_params, params_file)
    # params_file.close()

    #   Final RGB and depth equirectangular image
    equirect_rgb = np.moveaxis(equirect_3D_points_rgb[3:6].ravel().reshape(equirect_size), 0, -1)
    equirect_depth = equirect_3D_points_rgb[6].reshape(equirect_size[1:])

    return np.array(RGB_sub_images), np.array(depth_sub_images), xx.shape


def get_fisheyeFOV(img, model):
    #   Get (top center vs bottom center) and (left center vs right center) pixels to calculate FOV
    top_center_pt = np.array([0, 0])  
    bot_center_pt = np.array([0, img.shape[0]])
    left_center_pt = np.array([0, 0])
    right_center_pt = np.array([img.shape[1], 0])

    top_center_world = cam2world(np.array([top_center_pt]), model)[0]
    bot_center_world = cam2world(np.array([bot_center_pt]), model)[0]
    left_center_world = cam2world(np.array([left_center_pt]), model)[0]
    right_center_world = cam2world(np.array([right_center_pt]), model)[0]

    v_fov = np.degrees(np.arccos(np.clip(np.dot(bot_center_world, top_center_world), -1.0, 1.0)))
    h_fov = np.degrees(np.arccos(np.clip(np.dot(right_center_world, left_center_world), -1.0, 1.0)))

    return v_fov*2, h_fov*2


def plot_img_array(camera_array_size, sub_images):
    fig, axes = plt.subplots(nrows=camera_array_size[0], ncols=camera_array_size[1])
    for i in range(0, camera_array_size[0]):
        for j in range(0, camera_array_size[1]):
            idx = i*camera_array_size[1]+j
            axes[i, j].axis("off")
            axes[i, j].imshow(sub_images[idx])
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


def cam2world_single(point2D, model):
    """Converts a single camera pixel (2D) to world-space ray (3D).
    NB: Straightforward translation of Scaramuzza's C++ code."""

    c = model['intrinsics']['stretch_matrix'][0]
    d = model['intrinsics']['stretch_matrix'][1]
    e = model['intrinsics']['stretch_matrix'][2]
    invdet = 1. / (c - d * e)  # 1/det(A), where A = [c,d;e,1] as in the Matlab file

    xc = model['intrinsics']['distortion_center'][1]
    yc = model['intrinsics']['distortion_center'][0]
    xp = invdet * ((point2D[0] - xc) - d * (point2D[1] - yc) )
    yp = invdet * (-e * (point2D[0] - xc) + c * (point2D[1] - yc) )

    pol = model['intrinsics']['mapping_coefficients']
    r   = np.sqrt(xp * xp + yp * yp)  # distance [pixels] of  the point from the image center
    zp  = pol[0]
    r_i = 1

    for i in range(1, len(pol)):
        r_i *= r
        zp  += r_i * pol[i]

    ## Normalise to unit norm.
    invnorm = 1. / np.sqrt(xp * xp + yp * yp + zp * zp)
    point3D = invnorm * np.array([xp, yp, zp])

    return point3D


def world2cam_single(point3D, model):
    """Converts a single world-space ray (3D) to a camera pixel (2D).
    NB: Straightforward translation of Scaramuzza's C++ code."""

    xc = model['intrinsics']['distortion_center'][0]
    yc = model['intrinsics']['distortion_center'][1]

    norm = np.sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1])

    if norm != 0:
        theta = np.arctan(point3D[2] / norm)

        invpol = np.array(model['intrinsics']['fast_poly'])[::-1]
        invnorm = 1.0 / norm
        t  = theta
        rho = invpol[0]
        t_i = 1

        for i in range(1, len(invpol)):
            t_i *= t
            rho += t_i * invpol[i]

        x = point3D[0] * invnorm * rho
        y = point3D[1] * invnorm * rho

        c = model['intrinsics']['stretch_matrix'][0]
        d = model['intrinsics']['stretch_matrix'][1]
        e = model['intrinsics']['stretch_matrix'][2]
        point2D = np.zeros(2)
        point2D[0] = x * c + y * d + xc
        point2D[1] = x * e + y + yc

    else:
        point2D = np.array([xc, yc])

    return point2D


def equirect_world2cam(matrix3d, im_size=(1000, 1000)):
    # FIXME This is not tested and wrong. But at the moment I dont need it
    # implemented from https://ttic.uchicago.edu/~rurtasun/courses/CV/lecture08.pdf slide 32. What is s????????
    # s might be size of the image. [height, width] maybe square?
    # sphere_xy_center = (matrix3d[0, :] == 0) & (matrix3d[1, :] == 0)
    # fix_x = np.where(matrix3d[0, :] == 0)
    #
    # xp = im_size[1] * np.clip(np.arctan(matrix3d[1, :] / matrix3d[0, :])/(np.pi*0.5), -1.0, 1.0) + im_size[1]
    # xp[fix_x] = im_size[1]
    # yp = np.clip(im_size[0] * (matrix3d[2, :] / np.linalg.norm(matrix3d[[0, 1], :], axis=0)), -im_size[0], im_size[0])
    # yp[sphere_xy_center] = im_size[0]
    #
    #
    # points_2D = np.zeros((2, xp.shape[0]))
    # points_2D[0, :] = xp
    # points_2D[1, :] = yp

    #   Other implementation
    matrix3d = np.divide(matrix3d, np.tile(np.linalg.norm(matrix3d, axis=0), (3, 1)))
    lon = np.arctan2(matrix3d[1, :], matrix3d[0, :])
    lat = np.arctan2(matrix3d[2, :], np.linalg.norm(matrix3d[[0, 1], :], axis=0))

    xp = (im_size[1]*0.5) * (lon / np.pi) + (im_size[1]*0.5)
    yp = (im_size[0]*0.5) * (2 * lat / np.pi) + (im_size[0]*0.5)

    points_2D = np.zeros((2, xp.shape[0]))
    points_2D[0, :] = xp
    points_2D[1, :] = yp

    return points_2D

    # Remap with new 2D equirectangular coords


def equirect_cam2world(im_size=(1000, 1000)):

    canvas_yy, canvas_xx = np.mgrid[0:im_size[0], 0:im_size[1]]
    canvas_2d = np.squeeze(np.dstack((canvas_xx.flatten(), canvas_yy.flatten()))).reshape(-1, 2).T

    theta = (2*np.pi*canvas_2d[0, :]/(im_size[1]))
    phi = (np.pi * canvas_2d[1, :] / (im_size[0]))

    Y = np.cos(phi)
    Z = np.sin(phi) * -np.cos(theta)
    X = np.sin(phi) * -np.sin(theta)

    Z[Z < 0] = np.nan   # Dont consider anything on the back of the cylinder

    # Scaramuzza original 3D coordinate system. Z points inwards
    #
    #   Z                                                           ^ Y
    #   *------>Y                                                   |
    #   |                                                           |
    #   |           Remember that we rotated the image 90º          |
    #   |           counter-clockwise therefore now               Z *-------->X
    #   v X         it changes to. Check graph in the right
    #

    points_3D = np.zeros((3, im_size[0]*im_size[1]))
    points_3D[0, :] = X
    points_3D[1, :] = Y[::-1]
    points_3D[2, :] = Z

    unit_sphere = np.divide(points_3D, np.tile(np.linalg.norm(points_3D, axis=0), (3, 1)))

    return points_3D, unit_sphere


def generate_camera_orientation(hfov_fisheye, vfov_fisheye, hfov_pinhole, vfov_pinhole, horizontal_size, vertical_size, padding=0):
    """ Generate the camera orientation from the FOV and grid size

    :param vertical_size: camera orientation list, [3, horizontal_size * vertical_size], each row are rotation along the x, y and z axises.
    :type vertical_size: numpy
    :param padding: the padding area angle (degree) of the fisheye image along the edge of image.
    :type padding: float
    """
    hfov_interval = (hfov_fisheye - hfov_pinhole - padding * 2) / (horizontal_size - 1)
    vfov_interval = (vfov_fisheye - vfov_pinhole - padding * 2) / (vertical_size - 1)

    h_index = (np.linspace(0, horizontal_size, horizontal_size, endpoint=False) - (horizontal_size - 1) / 2.0) * hfov_interval
    v_index = (np.linspace(0, vertical_size, vertical_size, endpoint=False) - (vertical_size - 1) / 2.0) * vfov_interval
    x_rotation, y_rotation = np.meshgrid(h_index, v_index)  # the camere orientation

    # compute the overlap of perspective images.
    overlap_area_h = h_index[0] + hfov_pinhole / 2.0 - (h_index[1] - hfov_pinhole / 2.0)
    log.debug("the horizontal overlap angle is {}".format(overlap_area_h))
    overlap_area_v = v_index[0] + vfov_pinhole / 2.0 - (v_index[1] - vfov_pinhole / 2.0)
    log.debug("the vertical overlap angle is {}".format(overlap_area_v))

    z_rotation = np.zeros(x_rotation.shape, float)
    xyz_rotation_array = np.stack((x_rotation, y_rotation, z_rotation), axis=0)
    xyz_rotation_array = xyz_rotation_array.reshape([3, horizontal_size * vertical_size])
    return xyz_rotation_array


def get_perspective_camera_parameters(hfov, vfov, image_width, image_height, camera_direction):
    """create the pinhole camera parameters.

    :param hfov: the horizental field of view, radian.
    :type hfov: float
    :param vfov: the verical field of view, radian.
    :type vfov: float
    :param image_width: pin-hole image width.
    :type image_width: float
    :param image_height: pin-hole image height.
    :type image_height: float
    :param camera_direction: the camera direction array, the 3 columns are angle rotate along the x,y,z axises, size is [n,3].
    :type camera_direction: numpy
    :return: camera intrinsic and extrinsic parameters, the array is row mojor.
    :rtype: dict
    """  
    fx = 0.5*image_width / np.tan(np.radians(0.5*hfov))
    fy = 0.5*image_height / np.tan(np.radians(0.5*vfov))

    rotation = R.from_euler("zyx", camera_direction[:,[2,1,0]], degrees=True)
    rotation_mat_list = rotation.as_matrix()

    cx = image_width / 2.0
    cy = image_height / 2.0
    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

    params_list= []
    for rotation_mat in rotation_mat_list:
        params = {'rotation': rotation_mat,
                'translation': np.array([0, 0, 0]),
                'intrinsics': {
                    'image_width': image_width,
                    'image_height':image_height,
                    'focal_length_x': fx,
                    'focal_length_y': fy,
                    'principal_point': [cx, cy],
                    'matrix': intrinsic_matrix}
                }

        params_list.append(params)

    return params_list


def getVirtualCameraMatrix(viewfield, size, fisheye_cam, x_angle=0, y_angle=0, z_angle=0):
    """Create virtual camera intrinsic parameters. 

    Use the camera fov and image size to figure out the camera intrinsic parameters.

    TODO suport rectangle pinthole image

    :param viewfield: the camera's horizontal fov (field of view)
    :type viewfield: list or float
    :param size: the camera's horizontal fov (field of view)
    :type size: list
    :param x_angle: x_anlge, defaults to 0
    :type x_angle: int, optional
    :param y_angle: x_angle, defaults to 0
    :type y_angle: int, optional
    :param z_angle: z_angle, defaults to 0
    :type z_angle: int, optional
    :return: camera intrinsic
    :rtype: dict
    """  
    f = 0.5*size[1] / np.tan(np.radians(0.5*viewfield))

    intrinsic_matrix = np.array([[f, 0, size[1]/2],
                                 [0, f, size[0]/2],
                                 [0, 0, 1]])

    rmat_x = np.array([[1,             0,                      0],
                       [0, np.cos(np.radians(x_angle)), -np.sin(np.radians(x_angle))],
                       [0, np.sin(np.radians(x_angle)), np.cos(np.radians(x_angle))]])

    rmat_y = np.array([[np.cos(np.radians(y_angle)), 0, np.sin(np.radians(y_angle))],
                       [0,                           1,                 0],
                       [-np.sin(np.radians(y_angle)), 0, np.cos(np.radians(y_angle))]])

    rmat_z = np.array([[np.cos(np.radians(z_angle)), -np.sin(np.radians(z_angle)), 0],
                       [np.sin(np.radians(z_angle)), np.cos(np.radians(z_angle)),  0],
                       [0,             0,                      1]])

    params = {'rotation': rmat_z @ rmat_y @ rmat_x,
              'translation': np.array([0, 0, 0]),
              'intrinsics': {
                  'image_size': size,
                  'focal_length': f,
                  'principal_point': [size[0]/2, size[1]/2],
                  'matrix': intrinsic_matrix}}

    params_json = {'rotation': (rmat_z @ rmat_y @ rmat_x).tolist(),
                   'translation': [0, 0, 0],
                   'intrinsics': {
                        'image_size': size.tolist(),
                        'focal_length': f.tolist(),
                        'principal_point': [size[0]/2, size[1]/2],
                        'matrix': intrinsic_matrix.tolist()}}

    return params, params_json
