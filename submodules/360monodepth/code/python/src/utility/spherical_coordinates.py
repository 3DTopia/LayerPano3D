import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False

def great_circle_distance_uv(points_1_theta, points_1_phi, points_2_theta, points_2_phi, radius=1):
    """
    @see great_circle_distance (haversine distances )
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html

    :param points_1_theta: theta in radians
    :type points_1_theta : numpy
    :param points_1_phi: phi in radians
    :type points_1_phi : numpy
    :param points_2_theta: radians
    :type points_2_theta: float
    :param points_2_phi: radians
    :type points_2_phi: float
    :return: The geodestic distance from point ot tangent point.
    :rtype: numpy
    """
    delta_theta = points_2_theta - points_1_theta
    delta_phi = points_2_phi - points_1_phi
    a = np.sin(delta_phi * 0.5) ** 2 + np.cos(points_1_phi) * np.cos(points_2_phi) * np.sin(delta_theta * 0.5) ** 2
    central_angle_delta = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    if np.isnan(central_angle_delta).any():
        log.warn("the circle angle have NAN")

    return np.abs(radius * central_angle_delta)


def erp_pixel_modulo_0(erp_points_list, image_height):
    """[summary]

    :param erp_points_list: The erp pixel list, [2, points_number]
    :type erp_points_list: numpy
    :param image_height: erp image height
    :type image_height: numpy
    """
    x = erp_points_list[0,:]
    y = erp_points_list[1,:]
    x, y = erp_pixel_modulo(x, y , image_height)
    return np.stack((x,y), axis=0)


def erp_pixel_modulo(x_arrray, y_array, image_height):
    """ Make x,y and ERP pixels coordinate system range.
    """
    image_width = 2 * image_height
    x_arrray_new = np.remainder(x_arrray + 0.5, image_width) - 0.5
    y_array_new = np.remainder(y_array + 0.5, image_height) - 0.5
    return x_arrray_new, y_array_new


def erp_sph_modulo(theta, phi):
    """Modulo of the spherical coordinate for the erp coordinate.
    """
    points_theta = np.remainder(theta + np.pi, 2 * np.pi) - np.pi
    points_phi = -(np.remainder(-phi + 0.5 * np.pi, np.pi) - 0.5 * np.pi)
    return points_theta, points_phi


def erp2sph(erp_points, erp_image_height=None, sph_modulo=False):
    """
    convert the point from erp image pixel location to spherical coordinate.
    The image center is spherical coordinate origin.

    :param erp_points: the point location in ERP image x∊[0, width-1], y∊[0, height-1] , size is [2, :]
    :type erp_points: numpy
    :param erp_image_height: ERP image's height, defaults to None
    :type erp_image_height: int, optional
    :param sph_modulo: if true, process the input points wrap around, .
    :type sph_modulo: bool
    :return: the spherical coordinate points, theta is in the range [-pi, +pi), and phi is in the range [-pi/2, pi/2)
    :rtype: numpy
    """
    # 0) the ERP image size
    if erp_image_height == None:
        height = np.shape(erp_points)[1]
        width = np.shape(erp_points)[2]

        if (height * 2) != width:
            log.error("the ERP image width {} is not two time of height {}".format(width, height))
    else:
        height = erp_image_height
        width = height * 2

    erp_points_x = erp_points[0]
    erp_points_y = erp_points[1]

    # 1) point location to theta and phi
    points_theta = erp_points_x * (2 * np.pi / width) + np.pi / width - np.pi
    points_phi = -(erp_points_y * (np.pi / height) + np.pi / height * 0.5) + 0.5 * np.pi

    if sph_modulo:
        points_theta, points_phi = erp_sph_modulo(points_theta, points_phi)

    points_theta = np.where(points_theta == np.pi,  -np.pi, points_theta)
    points_phi = np.where(points_phi == -0.5 * np.pi, 0.5 * np.pi, points_phi)

    return np.stack((points_theta, points_phi))


def sph2erp_0(sph_points, erp_image_height=None, sph_modulo=False):
    theta = sph_points[0, :]
    phi = sph_points[1, :]
    erp_x, erp_y = sph2erp(theta, phi, erp_image_height, sph_modulo)
    return np.stack((erp_x, erp_y), axis=0)


def sph2erp(theta, phi, erp_image_height, sph_modulo=False):
    """ 
    Transform the spherical coordinate location to ERP image pixel location.

    :param theta: longitude is radian
    :type theta: numpy
    :param phi: latitude is radian
    :type phi: numpy
    :param image_height: the height of the ERP image. the image width is 2 times of image height
    :type image_height: [type]
    :param sph_modulo: if yes process the wrap around case, if no do not.
    :type sph_modulo: bool, optional
    :return: the pixel location in the ERP image.
    :rtype: numpy
    """
    if sph_modulo:
        theta, phi = erp_sph_modulo(theta, phi)

    erp_image_width = 2 * erp_image_height
    erp_x = (theta + np.pi) / (2.0 * np.pi / erp_image_width) - 0.5
    erp_y = (-phi + 0.5 * np.pi) / (np.pi / erp_image_height) - 0.5
    return erp_x, erp_y


def car2sph(points_car, min_radius=1e-10):
    """
    Transform the 3D point from cartesian to unit spherical coordinate.

    :param points_car: The 3D point array, is [point_number, 3], first column is x, second is y, third is z
    :type points_car: numpy
    :return: the points spherical coordinate, (theta, phi)
    :rtype: numpy
    """
    radius = np.linalg.norm(points_car, axis=1)

    valid_list = radius > min_radius  # set the 0 radius to origin.

    theta = np.zeros((points_car.shape[0]), float)
    theta[valid_list] = np.arctan2(points_car[:, 0][valid_list], points_car[:, 2][valid_list])

    phi = np.zeros((points_car.shape[0]), float)
    phi[valid_list] = -np.arcsin(np.divide(points_car[:, 1][valid_list], radius[valid_list]))

    return np.stack((theta, phi), axis=1)


def sph2car(theta, phi, radius=1.0):
    """
    Transform the spherical coordinate to cartesian 3D point.

    :param theta: longitude
    :type theta: numpy
    :param phi: latitude
    :type phi: numpy
    :param radius: the radius of projection sphere
    :type radius: float
    :return: +x right, +y down, +z is froward, shape is [3, point_number]
    :rtype: numpy
    """
    # points_cartesian_3d = np.array.zeros((theta.shape[0],3),float)
    x = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.cos(phi) * np.cos(theta)
    y = -radius * np.sin(phi)

    return np.stack((x, y, z), axis=0)
