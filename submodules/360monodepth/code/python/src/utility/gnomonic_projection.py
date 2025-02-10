
import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

"""
Implement the Gnomonic projection (forward and reverse projection).
Reference:
[1]: https://mathworld.wolfram.com/GnomonicProjection.html
"""


def inside_polygon_2d(points_list, polygon_points, on_line=False, eps=1e-4):
    """ Test the points inside the polygon. 
    Implement 2D PIP (Point Inside a Polygon).
    
    :param points_list: The points locations numpy array whose size is [point_numb, 2]. The point storage list is as [[x_1, y_1], [x_2, y_2],...[x_n, y_n]].
    :type points_list: numpy
    :param polygon_points: The clock-wise points sequence. The storage is the same as points_list.
    :type polygon_points: numpy
    :param on_line: The inside point including the boundary points, if True. defaults to False
    :type on_line: bool, optional
    :param eps: Use the set the polygon's line width. The distance between two pixel. defaults to 1e-4
    :type eps: float, optional
    :return: A numpy Boolean array, True is inside the polygon, False is outside.
    :rtype: numpy
    """
    point_inside = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point in the polygon
    online_index = np.full(np.shape(points_list)[0], False, dtype=bool)  # the point on the polygon lines

    points_x = points_list[:, 0]
    points_y = points_list[:, 1]

    def GREATER(a, b): return a >= b
    def LESS(a, b): return a <= b

    # try each line segment
    for index in range(np.shape(polygon_points)[0]):
        polygon_1_x = polygon_points[index][0]
        polygon_1_y = polygon_points[index][1]

        polygon_2_x = polygon_points[(index + 1) % len(polygon_points)][0]
        polygon_2_y = polygon_points[(index + 1) % len(polygon_points)][1]

        # exist points on the available XY range
        test_result = np.logical_and(GREATER(points_y, min(polygon_1_y, polygon_2_y)), LESS(points_y, max(polygon_1_y, polygon_2_y)))
        test_result = np.logical_and(test_result, LESS(points_x, max(polygon_1_x, polygon_2_x)))
        if not test_result.any():
            continue

        # get the intersection points
        if LESS(abs(polygon_1_y - polygon_2_y), eps):
            test_result = np.logical_and(test_result, GREATER(points_x, min(polygon_1_x, polygon_2_x)))
            intersect_points_x = points_x[test_result]
        else:
            intersect_points_x = (points_y[test_result] - polygon_1_y) * \
                (polygon_2_x - polygon_1_x) / (polygon_2_y - polygon_1_y) + polygon_1_x

        # the points on the line
        on_line_list = LESS(abs(points_x[test_result] - intersect_points_x), eps)
        if on_line_list.any():
            online_index[test_result] = np.logical_or(online_index[test_result], on_line_list)

        # the point on the left of the line
        if LESS(points_x[test_result], intersect_points_x).any():
            test_result[test_result] = np.logical_and(test_result[test_result], LESS(points_x[test_result], intersect_points_x))
            point_inside[test_result] = np.logical_not(point_inside[test_result])

    if on_line:
        return np.logical_or(point_inside, online_index).reshape(np.shape(points_list[:, 0]))
    else:
        return np.logical_and(point_inside, np.logical_not(online_index)).reshape(np.shape(points_list[:, 0]))


def gnomonic_projection(theta, phi, theta_0, phi_0):
    """ Gnomonic projection.
    Convet point form the spherical coordinate to tangent image's coordinate.
        https://mathworld.wolfram.com/GnomonicProjection.html

    :param theta: spherical coordinate's longitude.
    :type theta: numpy
    :param phi: spherical coordinate's latitude.
    :type phi: numpy
    :param theta_0: the tangent point's longitude of gnomonic projection.
    :type theta_0: float
    :param phi_0: the tangent point's latitude of gnomonic projection.
    :type phi_0: float
    :return: The gnomonic coordinate normalized coordinate.
    :rtype: numpy
    """
    cos_c = np.sin(phi_0) * np.sin(phi) + np.cos(phi_0) * np.cos(phi) * np.cos(theta - theta_0)

    # get cos_c's zero element index
    zeros_index = cos_c == 0
    if np.any(zeros_index):
        cos_c[zeros_index] = np.finfo(float).eps

    x = np.cos(phi) * np.sin(theta - theta_0) / cos_c
    y = (np.cos(phi_0) * np.sin(phi) - np.sin(phi_0) * np.cos(phi) * np.cos(theta - theta_0)) / cos_c

    if np.any(zeros_index):
        x[zeros_index] = 0
        y[zeros_index] = 0

    return x, y


def reverse_gnomonic_projection(x, y, lambda_0, phi_1):
    """ Reverse gnomonic projection.
    Convert the gnomonic nomalized coordinate to spherical coordinate.

    :param x: the gnomonic plane coordinate x.
    :type x: numpy 
    :param y: the gnomonic plane coordinate y.
    :type y: numpy
    :param theta_0: the gnomonic projection tangent point's longitude.
    :type theta_0: float
    :param phi_0: the gnomonic projection tangent point's latitude f .
    :type phi_0: float
    :return: the point array's spherical coordinate location. the longitude range is continuous and exceed the range [-pi, +pi]
    :rtype: numpy
    """
    rho = np.sqrt(x**2 + y**2)

    # get rho's zero element index
    zeros_index = rho == 0
    if np.any(zeros_index):
        rho[zeros_index] = np.finfo(float).eps

    c = np.arctan2(rho, 1)
    phi_ = np.arcsin(np.cos(c) * np.sin(phi_1) + (y * np.sin(c) * np.cos(phi_1)) / rho)
    lambda_ = lambda_0 + np.arctan2(x * np.sin(c), rho * np.cos(phi_1) * np.cos(c) - y * np.sin(phi_1) * np.sin(c))

    if np.any(zeros_index):
        phi_[zeros_index] = phi_1
        lambda_[zeros_index] = lambda_0

    return lambda_, phi_


def gnomonic2pixel(coord_gnom_x, coord_gnom_y,
                   padding_size,
                   tangent_image_width, tangent_image_height=None,
                   coord_gnom_xy_range=None):
    """Transform the tangent image's gnomonic coordinate to tangent image pixel coordinate.

    The tangent image gnomonic x is right, y is up.
    The tangent image pixel coordinate is x is right, y is down.

    :param coord_gnom_x: tangent image's normalized x coordinate
    :type coord_gnom_x: numpy
    :param coord_gnom_y: tangent image's normalized y coordinate
    :type coord_gnom_y: numpy
    :param padding_size: in gnomonic coordinate system, padding outside to boundary, in most case it's 0.0
    :type padding_size: float
    :param tangent_image_width: the image width with padding
    :type tangent_image_width: float
    :param tangent_image_height: the image height with padding
    :type tangent_image_height: float
    :param coord_gnom_xy_range: the range of gnomonic coordinate, [x_min, x_max, y_min, y_max]. It's often [-1.0 - padding_size, +1.0 + padding_size, ]
    :type coord_gnom_xy_range: list
    :retrun: the pixel's location
    :rtype: numpy (int)
    """
    if tangent_image_height is None:
        tangent_image_height = tangent_image_width

    # the gnomonic coordinate range of tangent image
    if coord_gnom_xy_range is None:
        x_min = -1.0
        x_max = 1.0 
        y_min = -1.0 
        y_max = 1.0 
    else:
        x_min = coord_gnom_xy_range[0]
        x_max = coord_gnom_xy_range[1]
        y_min = coord_gnom_xy_range[2]
        y_max = coord_gnom_xy_range[3]

    if padding_size != 0.0 and coord_gnom_xy_range is not None:
        log.warning("set the padding size and gnomonic range at same time! Please double check!")

    # normailzed tangent image space --> tangent image space
    # TODO check add the padding whether necessary
    gnomonic2image_width_ratio = (tangent_image_width - 1.0) / (x_max - x_min + padding_size * 2.0)
    coord_pixel_x = (coord_gnom_x - x_min + padding_size) * gnomonic2image_width_ratio
    coord_pixel_x = (coord_pixel_x + 0.5).astype(int)

    gnomonic2image_height_ratio = (tangent_image_height - 1.0) / (y_max - y_min + padding_size * 2.0)
    coord_pixel_y = -(coord_gnom_y - y_max - padding_size) * gnomonic2image_height_ratio
    coord_pixel_y = (coord_pixel_y + 0.5).astype(int)

    return coord_pixel_x, coord_pixel_y


def pixel2gnomonic(coord_pixel_x, coord_pixel_y,  padding_size,
                   tangent_image_width, tangent_image_height=None,
                   coord_gnom_xy_range=None):
    """Transform the tangent image's from tangent image pixel coordinate to gnomonic coordinate.

    :param coord_pixel_x: tangent image's pixels x coordinate
    :type coord_pixel_x: numpy
    :param coord_pixel_y: tangent image's pixels y coordinate
    :type coord_pixel_y: numpy
    :param padding_size: in gnomonic coordinate system, padding outside to boundary
    :type padding_size: float
    :param tangent_image_width: the image size with padding
    :type tangent_image_width: numpy
    :param tangent_image_height: the image size with padding
    :type tangent_image_height: numpy
    :param coord_gnom_xy_range: the range of gnomonic coordinate, [x_min, x_max, y_min, y_max]. It desn't includes padding outside to boundary.
    :type coord_gnom_xy_range: list
    :retrun: the pixel's location 
    :rtype:
    """
    if tangent_image_height is None:
        tangent_image_height = tangent_image_width

    # the gnomonic coordinate range of tangent image
    if coord_gnom_xy_range is None:
        x_min = -1.0
        x_max = 1.0
        y_min = -1.0
        y_max = 1.0
    else:
        x_min = coord_gnom_xy_range[0]
        x_max = coord_gnom_xy_range[1]
        y_min = coord_gnom_xy_range[2]
        y_max = coord_gnom_xy_range[3]

    # tangent image space --> tangent normalized space
    gnomonic_size_x = abs(x_max - x_min)
    gnomonic2image_ratio_width = (tangent_image_width - 1.0) / (gnomonic_size_x + padding_size * 2.0)
    coord_gnom_x = coord_pixel_x / gnomonic2image_ratio_width + x_min - padding_size

    gnomonic_size_y = abs(y_max - y_min)
    gnomonic2image_ratio_height = (tangent_image_height - 1.0) / (gnomonic_size_y + padding_size * 2.0)
    coord_gnom_y = - coord_pixel_y / gnomonic2image_ratio_height + y_max + padding_size

    return coord_gnom_x, coord_gnom_y
