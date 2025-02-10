import copy
import numpy as np
from scipy import ndimage

import gnomonic_projection as gp
import spherical_coordinates as sc
import polygon

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Implement icosahedron projection and stitch with the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""


def get_icosahedron_parameters(triangle_index, padding_size=0.0):
    """
    Get icosahedron's tangent face's paramters.
    Get the tangent point theta and phi. Known as the theta_0 and phi_0.
    The erp image origin as top-left corner

    :return the tangent face's tangent point and 3 vertices's location.
    """
    # reference: https://en.wikipedia.org/wiki/Regular_icosahedron
    radius_circumscribed = np.sin(2 * np.pi / 5.0)
    radius_inscribed = np.sqrt(3) / 12.0 * (3 + np.sqrt(5))
    radius_midradius = np.cos(np.pi / 5.0)

    # the tangent point
    theta_0 = None
    phi_0 = None

    # the 3 points of tangent triangle in spherical coordinate
    triangle_point_00_theta = None
    triangle_point_00_phi = None
    triangle_point_01_theta = None
    triangle_point_01_phi = None
    triangle_point_02_theta = None
    triangle_point_02_phi = None

    # triangles' row/col range in the erp image
    # erp_image_row_start = None
    # erp_image_row_stop = None
    # erp_image_col_start = None
    # erp_image_col_stop = None

    theta_step = 2.0 * np.pi / 5.0
    # 1) the up 5 triangles
    if 0 <= triangle_index <= 4:
        # tangent point of inscribed spheric
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index * theta_step
        phi_0 = np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + np.pi * 2.0 / 5.0 / 2.0 + triangle_index * theta_step
        triangle_point_01_phi = np.pi / 2.0
        triangle_point_02_theta = -np.pi + (triangle_index + 1) * theta_step
        triangle_point_02_phi = np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = 0
        # erp_image_row_stop = (np.pi / 2 - np.arctan(0.5)) / np.pi
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp
        # erp_image_col_stop = 1.0 / 5.0 * (triangle_index_temp + 1)

    # 2) the middle 10 triangles
    # 2-0) middle-up triangles
    if 5 <= triangle_index <= 9:
        triangle_index_temp = triangle_index - 5
        # tangent point of inscribed spheric
        theta_0 = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        phi_0 = np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius)
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_00_phi = np.arctan(0.5)
        triangle_point_01_theta = -np.pi + (triangle_index_temp + 1) * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = -np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = (np.arccos(radius_inscribed / radius_circumscribed) + np.arccos(radius_inscribed / radius_midradius)) / np.pi
        # erp_image_row_stop = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        # erp_image_col_start = 1 / 5.0 * triangle_index_temp
        # erp_image_col_stop = 1 / 5.0 * (triangle_index_temp + 1)

    # 2-1) the middle-down triangles
    if 10 <= triangle_index <= 14:
        triangle_index_temp = triangle_index - 10
        # tangent point of inscribed spheric
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = -(np.pi / 2.0 - np.arccos(radius_inscribed / radius_circumscribed) - 2 * np.arccos(radius_inscribed / radius_midradius))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        if triangle_index_temp == 10:
            # cross the ERP image boundary
            triangle_point_00_theta = triangle_point_00_theta + 2 * np.pi
        triangle_point_01_theta = -np.pi + triangle_index_temp * theta_step
        triangle_point_01_phi = np.arctan(0.5)
        triangle_point_02_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.arctan(0.5)

        # # availied area of ERP image
        # erp_image_row_start = (np.pi / 2.0 - np.arctan(0.5)) / np.pi
        # erp_image_row_stop = (np.pi - np.arccos(radius_inscribed / radius_circumscribed) - np.arccos(radius_inscribed / radius_midradius)) / np.pi
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
        # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

    # 3) the down 5 triangles
    if 15 <= triangle_index <= 19:
        triangle_index_temp = triangle_index - 15
        # tangent point of inscribed spheric
        theta_0 = - np.pi + triangle_index_temp * theta_step
        phi_0 = - (np.pi / 2 - np.arccos(radius_inscribed / radius_circumscribed))
        # the tangent triangle points coordinate in tangent image
        triangle_point_00_theta = - np.pi - theta_step / 2.0 + triangle_index_temp * theta_step
        triangle_point_00_phi = -np.arctan(0.5)
        triangle_point_01_theta = - np.pi + theta_step / 2.0 + triangle_index_temp * theta_step
        # cross the ERP image boundary
        if triangle_index_temp == 15:
            triangle_point_01_theta = triangle_point_01_theta + 2 * np.pi
        triangle_point_01_phi = -np.arctan(0.5)
        triangle_point_02_theta = - np.pi + triangle_index_temp * theta_step
        triangle_point_02_phi = -np.pi / 2.0

        # # spherical coordinate (0,0) is in the center of ERP image
        # erp_image_row_start = (np.pi / 2.0 + np.arctan(0.5)) / np.pi
        # erp_image_row_stop = 1.0
        # erp_image_col_start = 1.0 / 5.0 * triangle_index_temp - 1.0 / 5.0 / 2.0
        # erp_image_col_stop = 1.0 / 5.0 * triangle_index_temp + 1.0 / 5.0 / 2.0

    tangent_point = [theta_0, phi_0]

    # the 3 vertices in tangent image's gnomonic coordinate
    triangle_points_tangent = []
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_00_theta, triangle_point_00_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_01_theta, triangle_point_01_phi, theta_0, phi_0))
    triangle_points_tangent.append(gp.gnomonic_projection(triangle_point_02_theta, triangle_point_02_phi, theta_0, phi_0))

    # pading the tangent image
    triangle_points_tangent_no_pading = copy.deepcopy(triangle_points_tangent)  # Needed for NN blending
    triangle_points_tangent_pading = polygon.enlarge_polygon(triangle_points_tangent, padding_size)

    # if padding_size != 0.0:
    triangle_points_tangent = copy.deepcopy(triangle_points_tangent_pading)

    # the points in spherical location
    triangle_points_sph = []
    for index in range(3):
        tri_pading_x, tri_pading_y = triangle_points_tangent_pading[index]
        triangle_point_theta, triangle_point_phi = gp.reverse_gnomonic_projection(tri_pading_x, tri_pading_y, theta_0, phi_0)
        triangle_points_sph.append([triangle_point_theta, triangle_point_phi])

    # compute bounding box of the face in spherical coordinate
    availied_sph_area = []
    availied_sph_area = np.array(copy.deepcopy(triangle_points_sph))
    triangle_points_tangent_pading = np.array(triangle_points_tangent_pading)
    point_insert_x = np.sort(triangle_points_tangent_pading[:, 0])[1]
    point_insert_y = np.sort(triangle_points_tangent_pading[:, 1])[1]
    availied_sph_area = np.append(availied_sph_area, [gp.reverse_gnomonic_projection(point_insert_x, point_insert_y, theta_0, phi_0)], axis=0)
    # the bounding box of the face with spherical coordinate
    availied_ERP_area_sph = []  # [min_longitude, max_longitude, min_latitude, max_lantitude]

    if 0 <= triangle_index <= 4:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.pi / 2.0)
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))  # the ERP Y axis direction as down
    elif 15 <= triangle_index <= 19:
        if padding_size > 0.0:
            availied_ERP_area_sph.append(-np.pi)
            availied_ERP_area_sph.append(np.pi)
        else:
            availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
            availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(-np.pi / 2.0)
    else:
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 0]))
        availied_ERP_area_sph.append(np.amax(availied_sph_area[:, 1]))
        availied_ERP_area_sph.append(np.amin(availied_sph_area[:, 1]))

    # else:
    #     triangle_points_sph.append([triangle_point_00_theta, triangle_point_00_theta])
    #     triangle_points_sph.append([triangle_point_01_theta, triangle_point_01_theta])
    #     triangle_points_sph.append([triangle_point_02_theta, triangle_point_02_theta])

    #     availied_ERP_area.append(erp_image_row_start)
    #     availied_ERP_area.append(erp_image_row_stop)
    #     availied_ERP_area.append(erp_image_col_start)
    #     availied_ERP_area.append(erp_image_col_stop)

    return {"tangent_point": tangent_point, "triangle_points_tangent": triangle_points_tangent,
            "triangle_points_sph": triangle_points_sph,
            "triangle_points_tangent_nopad": triangle_points_tangent_no_pading, "availied_ERP_area": availied_ERP_area_sph}


def erp2ico_image(erp_image, tangent_image_width, padding_size=0.0, full_face_image=False):
    """Project the equirectangular image to 20 triangle images.

    Project the equirectangular image to level-0 icosahedron.

    :param erp_image: the input equirectangular image, RGB image should be 3 channel [H,W,3], depth map' shape should be [H,W].
    :type erp_image: numpy array, [height, width, 3]
    :param tangent_image_width: the output triangle image size, defaults to 480
    :type tangent_image_width: int, optional
    :param padding_size: the output face image' padding size
    :type padding_size: float
    :param full_face_image: If yes project all pixels in the face image, no just project the pixels in the face triangle, defaults to False
    :type full_face_image: bool, optional
    :param depthmap_enable: if project depth map, return the each pixel's 3D points location in current camera coordinate system.
    :type depthmap_enable: bool
    :return: If erp is rgb image:
                1) a list contain 20 triangle images, the image is 4 channels, invalided pixel's alpha is 0, others is 1.
                2)
                3) None.
    
            If erp is depth map:
                1) a list contain 20 triangle images depth maps in tangent coordinate system.  The subimage's depth is 3D point could depth value.
                2) 
                3) 3D point cloud in tangent coordinate system. The pangent point cloud coordinate system is same as the world coordinate system. +y down, +x right and +z forward.
    :rtype: 
    """
    if full_face_image:
        log.debug("Generate rectangle tangent image.")
    else:
        log.debug("Generating triangle tangent image.")
        
    # ERP image size
    depthmap_enable = False
    if len(erp_image.shape) == 3:
        if np.shape(erp_image)[2] == 4:
            log.info("project ERP image is 4 channels RGB map")
            erp_image = erp_image[:, :, 0:3]
        log.info("project ERP image 3 channels RGB map")
    elif len(erp_image.shape) == 2:
        log.info("project ERP image is single channel depth map")
        erp_image = np.expand_dims(erp_image, axis=2)
        depthmap_enable = True

    erp_image_height = np.shape(erp_image)[0]
    erp_image_width = np.shape(erp_image)[1]
    channel_number = np.shape(erp_image)[2]

    if erp_image_width != erp_image_height * 2:
        raise Exception("the ERP image dimession is {}".format(np.shape(erp_image)))

    tangent_image_list = []
    tangent_image_gnomonic_xy = [] # [x[height, width], y[height, width]]
    tangent_3dpoints_list = []
    tangent_sphcoor_list = []

    tangent_image_height = int((tangent_image_width / 2.0) / np.tan(np.radians(30.0)) + 0.5)

    # generate tangent images
    for triangle_index in range(0, 20):
        log.debug("generate the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        tangent_triangle_vertices = np.array(triangle_param["triangle_points_tangent"])
        # the face gnomonic range in tangent space
        gnomonic_x_min = np.amin(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_x_max = np.amax(tangent_triangle_vertices[:, 0], axis=0)
        gnomonic_y_min = np.amin(tangent_triangle_vertices[:, 1], axis=0)
        gnomonic_y_max = np.amax(tangent_triangle_vertices[:, 1], axis=0)
        gnom_range_x = np.linspace(gnomonic_x_min, gnomonic_x_max, num=tangent_image_width, endpoint=True)
        gnom_range_y = np.linspace(gnomonic_y_max, gnomonic_y_min, num=tangent_image_height, endpoint=True)
        gnom_range_xv, gnom_range_yv = np.meshgrid(gnom_range_x, gnom_range_y)

        # the tangent triangle points coordinate in tangent image
        inside_list = np.full(gnom_range_xv.shape[:2], True, dtype=bool)
        if not full_face_image:
            gnom_range_xyv = np.stack((gnom_range_xv.flatten(), gnom_range_yv.flatten()), axis=1)
            pixel_eps = (gnomonic_x_max - gnomonic_x_min) / (tangent_image_width)
            inside_list = gp.inside_polygon_2d(gnom_range_xyv, tangent_triangle_vertices, on_line=True, eps=pixel_eps)
            inside_list = inside_list.reshape(gnom_range_xv.shape)

        # project to tangent image
        tangent_point = triangle_param["tangent_point"]
        tangent_triangle_theta_, tangent_triangle_phi_ = gp.reverse_gnomonic_projection(gnom_range_xv[inside_list], gnom_range_yv[inside_list], tangent_point[0], tangent_point[1])

        tangent_sphcoor_list.append(
            np.stack((tangent_triangle_theta_.reshape(gnom_range_xv.shape), tangent_triangle_phi_.reshape(gnom_range_xv.shape)))
        )

        # tansform from spherical coordinate to pixel location
        tangent_triangle_erp_pixel_x, tangent_triangle_erp_pixel_y = sc.sph2erp(tangent_triangle_theta_, tangent_triangle_phi_, erp_image_height, sph_modulo=True)

        # get the tangent image pixels value
        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        tangent_image_x, tangent_image_y = gp.gnomonic2pixel(gnom_range_xv[inside_list], gnom_range_yv[inside_list],
                                                             0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

        if depthmap_enable:
            tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], -1.0)
        else:
            tangent_image = np.full([tangent_image_height, tangent_image_width, channel_number], 255.0)
        for channel in range(0, np.shape(erp_image)[2]):
            tangent_image[tangent_image_y, tangent_image_x, channel] = \
                ndimage.map_coordinates(erp_image[:, :, channel], [tangent_triangle_erp_pixel_y, tangent_triangle_erp_pixel_x], order=1, mode='wrap', cval=255.0)

        # if the ERP image is depth map, get camera coordinate system 3d points
        tangent_3dpoints = None
        if depthmap_enable:
            # convert the spherical depth map value to tangent image coordinate depth value  
            center2pixel_length = np.sqrt(np.square(gnom_range_xv[inside_list])  + np.square(gnom_range_yv[inside_list]) + np.ones_like(gnom_range_yv[inside_list]))
            center2pixel_length = center2pixel_length.reshape((tangent_image_height, tangent_image_width, channel_number))
            tangent_3dpoints_z = np.divide(tangent_image , center2pixel_length)
            tangent_image = tangent_3dpoints_z

            # get x and y
            tangent_3dpoints_x = np.multiply(tangent_3dpoints_z , gnom_range_xv[inside_list].reshape((tangent_image_height, tangent_image_width, channel_number)))
            tangent_3dpoints_y = np.multiply(tangent_3dpoints_z , gnom_range_yv[inside_list].reshape((tangent_image_height, tangent_image_width, channel_number)))
            tangent_3dpoints = np.concatenate([tangent_3dpoints_x, -tangent_3dpoints_y, tangent_3dpoints_z], axis =2)
            
        # set the pixels outside the boundary to transparent
        # tangent_image[:, :, 3] = 0
        # tangent_image[tangent_image_y, tangent_image_x, 3] = 255
        tangent_image_list.append(tangent_image)
        tangent_3dpoints_list.append(tangent_3dpoints)

    # get the tangent image's gnomonic coordinate
    tangent_image_gnomonic_x = gnom_range_xv[inside_list].reshape((tangent_image_height, tangent_image_width))
    tangent_image_gnomonic_xy.append(tangent_image_gnomonic_x)
    tangent_image_gnomonic_y = gnom_range_yv[inside_list].reshape((tangent_image_height, tangent_image_width))
    tangent_image_gnomonic_xy.append(tangent_image_gnomonic_y)

    return tangent_image_list, tangent_sphcoor_list, [tangent_3dpoints_list, tangent_image_gnomonic_xy]


def ico2erp_image(tangent_images, erp_image_height, padding_size=0.0, blender_method=None):
    """Stitch the level-0 icosahedron's tangent image to ERP image.

    blender_method:
        - None: just sample the triangle area;
        - Mean: the mean value on the overlap area.

    TODO there are seam on the stitched erp image.

    :param tangent_images: 20 tangent images in order.
    :type tangent_images: a list of numpy
    :param erp_image_height: the output erp image's height.
    :type erp_image_height: int
    :param padding_size: the face image's padding size
    :type padding_size: float
    :param blender_method: the method used to blend sub-images. 
    :type blender_method: str
    :return: the stitched ERP image
    :type numpy
    """
    if len(tangent_images) != 20:
        log.error("The tangent's images triangle number is {}.".format(len(tangent_images)))

    if len(tangent_images[0].shape) == 3:
        images_channels_number = tangent_images[0].shape[2]
        if images_channels_number == 4:
            log.debug("the face image is RGBA image, convert the output to RGB image.")
            images_channels_number = 3
    elif len(tangent_images[0].shape) == 2:
        log.info("project single channel disp or depth map")
        images_channels_number = 1

    erp_image_width = erp_image_height * 2
    erp_image = np.full([erp_image_height, erp_image_width, images_channels_number], 0, np.float64)

    tangent_image_height = tangent_images[0].shape[0]
    tangent_image_width = tangent_images[0].shape[1]

    erp_weight_mat = np.zeros((erp_image_height, erp_image_width), dtype=np.float64)
    # stitch all tangnet images to ERP image
    for triangle_index in range(0, 20):
        log.debug("stitch the tangent image {}".format(triangle_index))
        triangle_param = get_icosahedron_parameters(triangle_index, padding_size)

        # 1) get all tangent triangle's available pixels coordinate
        availied_ERP_area = triangle_param["availied_ERP_area"]
        erp_image_col_start, erp_image_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2], erp_image_height, sph_modulo=False)
        erp_image_col_stop, erp_image_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3], erp_image_height, sph_modulo=False)

        # process the image boundary
        erp_image_col_start = int(erp_image_col_start) if int(erp_image_col_start) > 0 else int(erp_image_col_start - 0.5)
        erp_image_col_stop = int(erp_image_col_stop + 0.5) if int(erp_image_col_stop) > 0 else int(erp_image_col_stop)
        erp_image_row_start = int(erp_image_row_start) if int(erp_image_row_start) > 0 else int(erp_image_row_start - 0.5)
        erp_image_row_stop = int(erp_image_row_stop + 0.5) if int(erp_image_row_stop) > 0 else int(erp_image_row_stop)

        triangle_x_range = np.linspace(erp_image_col_start, erp_image_col_stop, erp_image_col_stop - erp_image_col_start + 1)
        triangle_y_range = np.linspace(erp_image_row_start, erp_image_row_stop, erp_image_row_stop - erp_image_row_start + 1)
        triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)
        # process the wrap around
        triangle_xv = np.remainder(triangle_xv, erp_image_width)
        triangle_yv = np.remainder(triangle_yv, erp_image_height)

        # 2) sample the pixel value from tanget image
        # project spherical coordinate to tangent plane
        spherical_uv = sc.erp2sph([triangle_xv, triangle_yv], erp_image_height=erp_image_height, sph_modulo=False)
        theta_0 = triangle_param["tangent_point"][0]
        phi_0 = triangle_param["tangent_point"][1]
        tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], theta_0, phi_0)

        # the pixels in the tangent triangle
        triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])
        gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
        gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
        gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
        gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)

        tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
        pixel_eps = abs(tangent_xv[0, 0] - tangent_xv[0, 1]) / (2 * tangent_image_width)

        if len(tangent_images[0].shape) == 2:
            tangent_images_subimage = np.expand_dims(tangent_images[triangle_index], axis=2)
        else:
            tangent_images_subimage = tangent_images[triangle_index]

        if blender_method is None:
            available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                         triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

            # the tangent available gnomonic coordinate sample the pixel from the tangent image
            tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                       0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

            for channel in range(0, images_channels_number):
                erp_image[triangle_yv[available_pixels_list].astype(int), triangle_xv[available_pixels_list].astype(int), channel] = \
                    ndimage.map_coordinates(tangent_images_subimage[:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
        elif blender_method == "mean":
            triangle_points_tangent = [[gnomonic_x_min, gnomonic_y_max],
                                       [gnomonic_x_max, gnomonic_y_max],
                                       [gnomonic_x_max, gnomonic_y_min],
                                       [gnomonic_x_min, gnomonic_y_min]]
            available_pixels_list = gp.inside_polygon_2d(np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                                                         triangle_points_tangent, on_line=True, eps=pixel_eps).reshape(tangent_xv.shape)

            tangent_xv, tangent_yv = gp.gnomonic2pixel(tangent_xv[available_pixels_list], tangent_yv[available_pixels_list],
                                                       0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)
            for channel in range(0, images_channels_number):
                erp_face_image = ndimage.map_coordinates(tangent_images_subimage[:, :, channel], [tangent_yv, tangent_xv], order=1, mode='constant', cval=255)
                erp_image[triangle_yv[available_pixels_list].astype(int), triangle_xv[available_pixels_list].astype(int), channel] += erp_face_image.astype(np.float64)

            face_weight_mat = np.ones(erp_face_image.shape, np.float64)
            erp_weight_mat[triangle_yv[available_pixels_list].astype(np.int64), triangle_xv[available_pixels_list].astype(np.int64)] += face_weight_mat

    # compute the final optical flow base on weight
    if blender_method == "mean":
        # erp_flow_weight_mat = np.full(erp_flow_weight_mat.shape, erp_flow_weight_mat.max(), float) # debug
        non_zero_weight_list = erp_weight_mat != 0
        if not np.all(non_zero_weight_list):
            log.warn("the optical flow weight matrix contain 0.")
        for channel_index in range(0, images_channels_number):
            erp_image[:, :, channel_index][non_zero_weight_list] = erp_image[:, :, channel_index][non_zero_weight_list] / erp_weight_mat[non_zero_weight_list]

    return erp_image
