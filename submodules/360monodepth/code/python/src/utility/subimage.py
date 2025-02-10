import spherical_coordinates
import projection_icosahedron as proj_ico
import gnomonic_projection as gp

from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageDraw
import numpy as np
from colorsys import hsv_to_rgb

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def draw_corresponding(src_image_data, tar_image_data, pixel_corresponding_array):
    """
    Visualized the pixel corresponding relationship.

    :param src_image_data: source image data
    :type src_image_data: numpy
    :param tar_image_data: target image data
    :type tar_image_data: numpy
    :param pixel_corresponding_array: the pixel corresponding from source, size is [4, :]
    :type pixel_corresponding_array: numpy 
    :return: the marked source and target image, and warped source image
    """
    # 0) prepare the canvas
    src_image_data_image = Image.fromarray(src_image_data.astype(np.uint8))
    # convert the gray image to rgb image
    if len(src_image_data.shape) == 2:
        src_image_data_image =  src_image_data_image.convert("RGB")
    src_image_draw = ImageDraw.Draw(src_image_data_image)

    tar_image_data_image = Image.fromarray(tar_image_data.astype(np.uint8))
    # convert the gray image to rgb image
    if len(src_image_data.shape) == 2:
        tar_image_data_image =  tar_image_data_image.convert("RGB")
    tar_image_draw = ImageDraw.Draw(tar_image_data_image)

    # the corresponding relationship is empty
    if pixel_corresponding_array.size == 0:
        log.warn("The pixel corresponding is empty!")
        src_warp_image = np.zeros_like(src_image_data_image)
        return np.asarray(src_image_data_image), np.asarray(tar_image_data_image), np.asarray(src_warp_image)

    # 1) draw corresponding pixels in source and target images
    index = 0
    eX, eY = 3, 3  # Size of Bounding Box for ellips
    minval = 0
    maxval = pixel_corresponding_array.shape[0] + 10

    for match_pair in pixel_corresponding_array:
        # if index % 100 == 0:
        #     print("plot corresponding {}".format(index))

        # r = index % 256
        # g = int(index % (256 ** 2) / 256)
        # b = int((index % (256 ** 3)) / 256 ** 2)

        h = (float(index-minval) / (maxval-minval)) * 120.0
        # Convert hsv color (h,1,1) to its rgb equivalent.
        # Note: hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
        r, g, b = (int(i * 255) for i in hsv_to_rgb(h/360.0, 1., 1.))

        y = match_pair[0]
        x = match_pair[1]
        bbox = (x - eX, y - eY, x + eX, y + eY)
        src_image_draw.ellipse(bbox, fill=(r, g, b))

        y = match_pair[2]
        x = match_pair[3]
        bbox = (x - eX, y - eY, x + eX, y + eY)
        tar_image_draw.ellipse(bbox, fill=(r, g, b))

        index += 1

    del src_image_draw
    del tar_image_draw

    src_image_data_image_np = np.asarray(src_image_data_image)
    tar_image_data_image_np = np.asarray(tar_image_data_image)

    # 2) warp src image
    src_warp_image = np.zeros(src_image_data.shape, src_image_data.dtype)
    pixel_corresponding_array_temp = pixel_corresponding_array.astype(int)
    src_y = pixel_corresponding_array_temp[:, 0]
    src_x = pixel_corresponding_array_temp[:, 1]
    tar_y = pixel_corresponding_array_temp[:, 2]
    tar_x = pixel_corresponding_array_temp[:, 3]
    from scipy import ndimage
    for channel in range(0, 3):
        src_warp_image[tar_y, tar_x, channel] = \
            ndimage.map_coordinates(src_image_data[:, :, channel], [src_y, src_x], order=1, mode='constant')

    return src_image_data_image_np, tar_image_data_image_np, src_warp_image.astype(np.uint8)


def tangent_image_resolution(erp_image_width, padding_size):
    """Get the the suggest tangent image resolution base on the FoV.

    :param erp_image_width: [description]
    :type erp_image_width: [type]
    :param padding_size: [description]
    :type padding_size: [type]
    :return: recommended tangent image size in pixel.
    :rtype: int
    """
    # camera intrinsic parameters
    ico_param_list = proj_ico.get_icosahedron_parameters(7, padding_size)
    triangle_points_tangent = ico_param_list["triangle_points_tangent"]
    # compute the tangent image resoution.
    tangent_points_x_min = np.amin(np.array(triangle_points_tangent)[:, 0])
    fov_h = np.abs(2 * np.arctan2(tangent_points_x_min, 1.0))
    tangent_image_width = erp_image_width * (fov_h / (2 * np.pi))
    tangent_image_height = 0.5 * tangent_image_width / np.tan(np.radians(30.0))
    return int(tangent_image_width + 0.5), int(tangent_image_height + 0.5)


def erp_ico_cam_intrparams(image_width, padding_size=0):
    """    
    Compuate the camera intrinsic parameters for 20 faces of icosahedron.
    It does not need camera parameters.

    :param image_width: Tangent image's width, the image height derive from image ratio.
    :type image_width: int
    :param padding_size: The tangent face padding size, defaults to 0
    :type padding_size: float, optional
    :return: 20 faces camera parameters.
    :rtype: list
    """
    # camera intrinsic parameters
    ico_param_list = proj_ico.get_icosahedron_parameters(7, padding_size)
    tangent_point = ico_param_list["tangent_point"]
    triangle_points_tangent = ico_param_list["triangle_points_tangent"]

    # use tangent plane
    tangent_points_x_min = np.amin(np.array(triangle_points_tangent)[:, 0])
    tangent_points_y_min = np.amin(np.array(triangle_points_tangent)[:, 1])
    tangent_points_y_max = np.amax(np.array(triangle_points_tangent)[:, 1])
    fov_v = np.abs(np.arctan2(tangent_points_y_min, 1.0)) + np.abs(np.arctan2(tangent_points_y_max, 1.0))
    fov_h = np.abs(2 * np.arctan2(tangent_points_x_min, 1.0))

    log.debug("Pin-hole camera fov_h: {}, fov_v: {}".format(np.degrees(fov_h), np.degrees(fov_v)))

    # image aspect ratio, the triangle is equilateral triangle
    image_height = 0.5 * image_width / np.tan(np.radians(30.0))
    fx = 0.5 * image_width / np.tan(fov_h * 0.5)
    fy = 0.5 * image_height / np.tan(fov_v * 0.5)

    cx = (image_width - 1) / 2.0
    # invert and upright triangle cy
    cy_invert = 0.5 * (image_width - 1.0) * np.tan(np.radians(30.0)) + 10.0
    cy_up = 0.5 * (image_width - 1.0) / np.sin(np.radians(60.0)) + 10.0

    subimage_cam_param_list = []
    for index in range(0, 20):
        # intrinsic parameters
        cy = None
        if 0 <= index <= 4:
            cy = cy_up
        elif 5 <= index <= 9:
            cy = cy_invert
        elif 10 <= index <= 14:
            cy = cy_up
        else:
            cy = cy_invert

        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]])

        # rotation
        ico_param_list = proj_ico.get_icosahedron_parameters(index, padding_size)
        tangent_point = ico_param_list["tangent_point"]
        # print(tangent_point)
        rot_y = tangent_point[0]
        rot_x = tangent_point[1]
        rotation = R.from_euler("zyx", [0.0, -rot_y, -rot_x], degrees=False)
        rotation_mat = rotation.as_matrix()

        params = {'rotation': rotation_mat,
                  'translation': np.array([0, 0, 0]),
                  'intrinsics': {
                      'image_width': image_width,
                      'image_height': image_height,
                      'focal_length_x': fx,
                      'focal_length_y': fy,
                      'principal_point': [cx, cy],
                      'matrix': intrinsic_matrix}
                  }

        subimage_cam_param_list.append(params)

    return subimage_cam_param_list


def erp_ico_pixel_corr(subimage_sphcoor, next_tangent_point, padding_size, tangent_image_width, tangent_image_height, tangent_triangle_vertices_gnom):
    """
    Get the corresponding point between two Ico's face.
    And return the pixel corresponding relationship, 
    [current_pixel_y, current_pixel_x, target_pixel_y, target_pixel_x]

    :param subimage_sphcoor: source subimage each pixel's spherical coordinate.
    :type subimage_sphcoor: numpy
    :param next_tangent_point: the target subimage's tangent point.
    :param tangent_image_width: 
    :param tangent_image_height:
    :param tangent_triangle_vertices_gnom: the 3 vertexes of tangent plane.
    :type tangent_triangle_vertices_gnom: list
    """
    gnom_x, gnom_y = gp.gnomonic_projection(subimage_sphcoor[0, :], subimage_sphcoor[1, :], next_tangent_point[0], next_tangent_point[1])

    # 1)remove the pixels on the another hemisphere, just use the pixel in the same hemisphere.
    gnomonic_x_min = np.amin(tangent_triangle_vertices_gnom[:, 0], axis=0)
    gnomonic_x_max = np.amax(tangent_triangle_vertices_gnom[:, 0], axis=0)
    gnomonic_y_min = np.amin(tangent_triangle_vertices_gnom[:, 1], axis=0)
    gnomonic_y_max = np.amax(tangent_triangle_vertices_gnom[:, 1], axis=0)
    tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
    gnom_image_x, gnom_image_y = gp.gnomonic2pixel(gnom_x, gnom_y, 0.0, tangent_image_width, tangent_image_height, tangent_gnomonic_range)

    points_2_theta = np.full(subimage_sphcoor[0, :].shape, next_tangent_point[0], np.float64)
    points_2_phi = np.full(subimage_sphcoor[0, :].shape, next_tangent_point[1], np.float64)
    central_angle_delta = \
        spherical_coordinates.great_circle_distance_uv(subimage_sphcoor[0, :], subimage_sphcoor[1, :], points_2_theta, points_2_phi, radius=1)

    valid_pixel_index = np.logical_and.reduce((
        gnom_image_x >= 0, gnom_image_x < tangent_image_width,
        gnom_image_y >= 0, gnom_image_y < tangent_image_height,
        np.abs(central_angle_delta) < 0.5 * np.pi))

    # 2) get the src ant tar's subimage pixel coordinate
    src_subimage_x = np.linspace(0, tangent_image_width, tangent_image_width, endpoint=False)
    src_subimage_y = np.linspace(0, tangent_image_height, tangent_image_height, endpoint=False)
    src_subimage_xv, src_subimage_yv = np.meshgrid(src_subimage_x, src_subimage_y)
    pixel_index_src = np.stack((src_subimage_yv[valid_pixel_index], src_subimage_xv[valid_pixel_index])).T
    pixel_index_tar = np.stack((gnom_image_y[valid_pixel_index], gnom_image_x[valid_pixel_index])).T

    # return pixels spherical coordinate 
    pixels_sph = subimage_sphcoor[:, valid_pixel_index]

    return np.hstack((pixel_index_src, pixel_index_tar)), pixels_sph


def erp_ico_proj(erp_image, padding_size, tangent_image_width, corr_downsample_factor, opt = None):
    """
    Using Icosahedron sample the ERP image to generate subimage, pixel corresponding and camera parameter.
    """
    if corr_downsample_factor != 1.0:
        log.info("Down sample the pixels corresponding, keep {}%.".format(corr_downsample_factor * 100))
        
    # 0) generate subimage
    subimage_list, subimage_sphcoor_list, _ = proj_ico.erp2ico_image(erp_image, tangent_image_width, padding_size, full_face_image=True)
    tangent_image_height = subimage_list[0].shape[0]

    # 1) compute current image overlap are with others subimage
    pixels_corr_dict = {}
    ico_param_list = []
    for index in range(0, len(subimage_list)):
        ico_param_list.append(proj_ico.get_icosahedron_parameters(index, padding_size))

    # set the matterport dataset flag
    if opt is None:
        matterport_hexagon_mask_enable = False
        matterport_hexagon_circumradius = -1
    else:
        matterport_hexagon_mask_enable = opt.dataset_matterport_hexagon_mask_enable
        erp_image_height = erp_image.shape[0]
        matterport_circle_phi = np.deg2rad(opt.dataset_matterport_blur_area_height * (180.0 / erp_image_height))
        matterport_hexagon_circumradius = np.tan(matterport_circle_phi)
        log.info(f"The image height is {erp_image_height}, margin height is {opt.dataset_matterport_blur_area_height}, circumradius is {matterport_hexagon_circumradius}")
        matterport_blurarea_shape = opt.dataset_matterport_blurarea_shape   # "hexagon",  "circle"


    # source subimage index
    def subimage_corr_fun(ico_param_list, pixels_corr_dict, subimage_index_src):
        subimage_sphcoor = subimage_sphcoor_list[subimage_index_src]

        # target subimage index, and compute the corresponding computing with multi-threads
        pixels_corr_dict_subimage = {}
        for subimage_index_tar in range(0, len(subimage_list)):
            # find the corresponding pixel from all subimage
            if subimage_index_src == subimage_index_tar:
                pixels_corr_dict_subimage[subimage_index_tar] = np.empty(shape=(0, 0))
                continue
            # compute the corresponding
            ico_param = ico_param_list[subimage_index_tar]
            heightbout_tangent_point = ico_param["tangent_point"]
            tangent_triangle_vertices_gnom = np.array(ico_param["triangle_points_tangent"])
            pixels_corr_src2tar, pixels_sph = erp_ico_pixel_corr(
                subimage_sphcoor, heightbout_tangent_point, padding_size, tangent_image_width, tangent_image_height, tangent_triangle_vertices_gnom)
            pixels_corr_src2tar = pixels_corr_src2tar.astype(np.float64)

            # remove the pixel at top and bottom
            if matterport_hexagon_mask_enable and \
                ((0 <= subimage_index_src <= 4 and 0 <= subimage_index_tar <= 4)
                 or (15 <= subimage_index_src <= 19 and 15 <= subimage_index_tar <= 19)):

                # 1) get the src and tar pixel coordinate on the top/bottom tangent image
                if 0 <= subimage_index_src <= 4:
                    hexagon_subimage_tangent_point_theta = 0.0
                    hexagon_subimage_tangent_point_phi = 0.5 * np.pi
                elif 15 <= subimage_index_tar <= 19:
                    hexagon_subimage_tangent_point_theta = 0.0
                    hexagon_subimage_tangent_point_phi = -0.5 * np.pi

                if matterport_blurarea_shape == "hexagon":
                    import polygon
                    # project to top or bottom tangent image
                    pixels_gnom_xv, pixels_gnom_yv = \
                        gp.gnomonic_projection(pixels_sph[0, :], pixels_sph[1, :], hexagon_subimage_tangent_point_theta, hexagon_subimage_tangent_point_phi)
                    pixels_corr_gnom = np.stack((pixels_gnom_xv, pixels_gnom_yv), axis=1)

                    # check if the pixels in the hexagon
                    hexagon_points_list = polygon.generate_hexagon(matterport_hexagon_circumradius)
                    inside_hexagon = gp.inside_polygon_2d(pixels_corr_gnom, hexagon_points_list)
                    outside_hexagon = np.logical_not(inside_hexagon)
                    pixels_corr_src2tar = pixels_corr_src2tar[outside_hexagon]
                elif matterport_blurarea_shape == "circle":
                    # import ipdb; ipdb.set_trace()
                    # outside_circle = np.logical_and(pixels_sph[1, :] <= np.pi-matterport_circle_phi, pixels_sph[1, :] >= matterport_circle_phi)
                    outside_circle = np.logical_and(pixels_sph[1, :] <= np.pi * 0.5 - matterport_circle_phi, pixels_sph[1, :] >= - np.pi * 0.5 + matterport_circle_phi)
                    pixels_corr_src2tar = pixels_corr_src2tar[outside_circle]

            # assign the value
            pixels_corr_dict_subimage[subimage_index_tar] = pixels_corr_src2tar

            # down-sample the pixel corresponding relationship
            if corr_downsample_factor != 1.0 and pixels_corr_dict_subimage[subimage_index_tar] is not None:
                corr_number = pixels_corr_dict_subimage[subimage_index_tar].shape[0]
                corr_index = np.linspace(0, corr_number -1, num = int(corr_number * corr_downsample_factor)).astype(int)
                corr_index = np.unique(corr_index)
                pixels_corr_dict_subimage[subimage_index_tar]  = pixels_corr_dict_subimage[subimage_index_tar][corr_index,:]
            
        pixels_corr_dict[subimage_index_src] = pixels_corr_dict_subimage
        return subimage_index_src

    # get the corresponding relationship with multi-thread
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=opt.dispalign_corr_thread_number) as executor:
        log.debug("Start generating imagepixels corresponding:")
        feature_list = []
        for subimage_index in range(0, len(subimage_list)):
            # log.debug("Generate image {} pixels corresponding: start ".format(subimage_index))
            feature_list.append(executor.submit(subimage_corr_fun, ico_param_list, pixels_corr_dict, subimage_index))

        for future in as_completed(feature_list):
            try:
                result = future.result()
            except Exception as exc:
                print('Computing subimage {} pixels corresponding error! Exception: {}' % (result, exc))
            else:
                log.debug("Generate image {} pixels corresponding: done ".format(result))

    # 2) camera parameters
    subimage_cam_param_list = erp_ico_cam_intrparams(tangent_image_width, padding_size)
    return subimage_list, subimage_cam_param_list, pixels_corr_dict


def erp_ico_stitch(subimage_list, erp_image_height, padding_size):
    """
    Stitch the Ico's subimages to ERP image.
    """
    # the 'mean' is use linear blend
    erp_image = proj_ico.ico2erp_image(subimage_list, erp_image_height, padding_size, "mean")
    return erp_image


def erp_ico_draw_corresponding(src_image_data, tar_image_data, pixel_corresponding_array,
                               src_image_output_path, tar_image_output_path):
    src_image_data_np = np.array(src_image_data)
    # 0) draw corresponding,
    if isinstance(src_image_data, np.ndarray):
        src_image_data_image = Image.fromarray(src_image_data)
    else:
        src_image_data_image = src_image_data
    if isinstance(tar_image_data, np.ndarray):
        tar_image_data_image = Image.fromarray(tar_image_data)
    else:
        tar_image_data_image = tar_image_data

    src_image_draw = ImageDraw.Draw(src_image_data_image)
    tar_image_draw = ImageDraw.Draw(tar_image_data_image)

    index = 0
    eX, eY = 3, 3  # Size of Bounding Box for ellips
    for match_pair in pixel_corresponding_array:
        # if index % 100 == 0:
        #     print("plot corresponding {}".format(index))

        y = match_pair[0]
        x = match_pair[1]
        bbox = (x - eX, y - eY, x + eX, y + eY)
        src_image_draw.ellipse(bbox, fill=(255, 0, 0))

        y = match_pair[2]
        x = match_pair[3]
        bbox = (x - eX, y - eY, x + eX, y + eY)
        tar_image_draw.ellipse(bbox, fill=(255, 0, 0))

        index += 1

    del src_image_draw
    del tar_image_draw

    src_image_data_image.save(src_image_output_path)
    print(src_image_output_path)
    tar_image_data_image.save(tar_image_output_path)
    print(tar_image_output_path)

    # 1) warp src image
    src_warp = np.zeros(src_image_data_np.shape, src_image_data_np.dtype)
    pixel_corresponding_array_temp = pixel_corresponding_array.astype(int)
    src_y = pixel_corresponding_array_temp[:, 0]
    src_x = pixel_corresponding_array_temp[:, 1]
    tar_y = pixel_corresponding_array_temp[:, 2]
    tar_x = pixel_corresponding_array_temp[:, 3]
    from scipy import ndimage
    for channel in range(0, 3):
        src_warp[tar_y, tar_x, channel] = \
            ndimage.map_coordinates(src_image_data_np[:, :, channel], [src_y, src_x], order=1, mode='constant')

    src_warp_image = Image.fromarray(src_warp)
    src_warp_image.save(src_image_output_path + "_features_warp.jpg")
