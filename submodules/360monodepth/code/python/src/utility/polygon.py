import numpy as np
from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


def find_intersection(p1,  p2,  p3,  p4):
    """Find the point of intersection between two line.
    Work on 3D plane.
    
    The two lines are p1 --> p2 and p3 --> p4.
    Reference:http://csharphelper.com/blog/2020/12/enlarge-a-polygon-that-has-colinear-vertices-in-c/

    :param p1: line 1's start point
    :type p1: list
    :param p2: line 1's end point
    :type p2: list
    :param p3: line 2's start point
    :type p3: list
    :param p4: line 2's end point
    :type p4: list
    :return: The intersection point of two line
    :rtype: list
    """
    # the segments
    dx12 = p2[0] - p1[0]
    dy12 = p2[1] - p1[1]
    dx34 = p4[0] - p3[0]
    dy34 = p4[1] - p3[1]

    denominator = (dy12 * dx34 - dx12 * dy34)
    if denominator == 0:
        # The two lines are parallel
        return None

    t1 = ((p1[0] - p3[0]) * dy34 + (p3[1] - p1[1]) * dx34) / denominator
    t2 = ((p3[0] - p1[0]) * dy12 + (p1[1] - p3[1]) * dx12) / -denominator

    # point of intersection.
    intersection = [p1[0] + dx12 * t1, p1[1] + dy12 * t1]
    return intersection


def is_clockwise(point_list):
    """Check whether the list is clockwise. 

    :param point_list: the point lists
    :type point_list: numpy 
    :return: yes if the points are clockwise, otherwise is no
    :rtype: bool
    """
    sum = 0
    for i in range(len(point_list)):
        cur = point_list[i]
        next = point_list[(i+1) % len(point_list)]
        sum += (next[0] - cur[0]) * (next[1] + cur[1])
    return sum > 0


def enlarge_polygon(old_points, offset):
    """Return points representing an enlarged polygon.

    Reference: http://csharphelper.com/blog/2016/01/enlarge-a-polygon-in-c/
    
    :param old_points: the polygon vertexes, and the points should be in clock wise
    :type: list [[x_1,y_1], [x_2, y_2]......]
    :param offset: the ratio of the polygon enlarged
    :type: float
    :return: the offset points
    :rtype: list
    """
    enlarged_points = []
    num_points = len(old_points)
    for j in range(num_points):
        # 0) find "out" side
        if not is_clockwise(old_points):
            log.error("the points list is not clockwise.")

        # the points before and after j.
        i = (j - 1)
        if i < 0:
            i += num_points
        k = (j + 1) % num_points

        # 1) Move the points by the offset.
        # the points of line parallel to ij
        v1 = np.array([old_points[j][0] - old_points[i][0], old_points[j][1] - old_points[i][1]], float)
        norm = np.linalg.norm(v1)
        v1 = v1 / norm * offset
        n1 = [-v1[1], v1[0]]
        pij1 = [old_points[i][0] + n1[0], old_points[i][1] + n1[1]]
        pij2 = [old_points[j][0] + n1[0], old_points[j][1] + n1[1]]

        # the points of line parallel to jk
        v2 = np.array([old_points[k][0] - old_points[j][0], old_points[k][1] - old_points[j][1]], float)
        norm = np.linalg.norm(v2)
        v2 = v2 / norm * offset
        n2 = [-v2[1], v2[0]]
        pjk1 = [old_points[j][0] + n2[0], old_points[j][1] + n2[1]]
        pjk2 = [old_points[k][0] + n2[0], old_points[k][1] + n2[1]]

        # 2) get the shifted lines ij and jk intersect
        lines_intersect = find_intersection(pij1, pij2, pjk1, pjk2)
        enlarged_points.append(lines_intersect)

    return enlarged_points


def generate_hexagon(circumradius=1.0, hexagon_type=0, draw_enable=False):
    """ Figure out the 6 vertexes of regular hexagon.
    The coordinate center is hexagon center.
    The original of coordinate at the center of hexagon.

    :param circumradius: The circumradius of the hexagon, defaults to 1.0
    :type circumradius: float, optional
    :param hexagon_type: the hexagon type, 0 the point is on the y-axis, 1 the vertexes on the x-axis, defaults to 0
    :type hexagon_type: int, optional
    :return: the six vertexes of hexagon.[6,2],  the 1st and 2nd is x,y respectively.
    :rtype: numpy 
    """
    vertex_list = np.zeros((6, 2), dtype=np.double)

    angle_interval = -60.0
    for idx in range(0, 6):
        if hexagon_type == 0:
            angle = angle_interval * idx
        elif hexagon_type == 1:
            angle = angle_interval * idx + 30.0
        else:
            log.error("Do not support hexagon type {}".format(hexagon_type))
        vertex_list[idx, 0] = np.cos(np.radians(angle)) * circumradius
        vertex_list[idx, 1] = np.sin(np.radians(angle)) * circumradius

    if draw_enable:
        from PIL import Image, ImageDraw
        image_width = circumradius * 3
        image_height = circumradius * 3
        offset_width = image_width * 0.5
        offset_height = image_height * 0.5
        image = Image.new('RGB', (image_height, image_width), 'white')
        draw = ImageDraw.Draw(image)
        vertex_list_ = np.zeros_like(vertex_list)
        vertex_list_[:, 0] = vertex_list[:, 0] + offset_width  # the origin at upper-left of image.
        vertex_list_[:, 1] = vertex_list[:, 1] + offset_height
        draw.polygon(tuple(map(tuple, vertex_list_)), outline='black', fill='red')
        image.show()

    return vertex_list


def isect_line_plane_3D(line_p0, line_p1, plane_point, plane_norm):
    """ Get the intersection point between plane and line.

    :param line_p0: a point on the line, shape is [3]
    :type line_p0: numpy
    :param line_p1: a point on the line, shape is [3]
    :type line_p1: numpy
    :param plane_point: a point on the 3D plane, shape is [3]
    :type plane_point: numpy
    :param plane_norm: a normal vector of the plane. shape is [3]
    :type plane_norm: numpy
    :return: the intersection point, shape is [3]
    :rtype: numpy
    """
    u = line_p0 - line_p1
    dot = np.dot(plane_norm, u)

    if abs(dot) > np.finfo(np.float32).eps:
        w = line_p0 - plane_point
        fac = - np.dot(plane_norm, w) / dot
        return line_p0 + u * fac
    else:
        return None


def triangle_bounding_rectangle_3D(head_point, edge_points):
    """ The 3D bounding rectangle from the 3D triangle's 3 vertices.

    :param head_point: The 3 vertices of the triangle, shape is [3], which is xyz.
    :type head_point: numpy
    :param edge_points: The 2 vertices of the edge, shape is [2, 3], each row is xyz
    :type edge_points: numpy
    :return: the 4 vertiec of the rectangle, shape is [4,3]
    :rtype: numpy
    """
    edge_point_A = edge_points[0, :]
    edge_point_B = edge_points[1, :]

    edge_points_mid = 0.5 * (edge_point_A + edge_point_B)
    mid_head_vec = head_point - edge_points_mid

    edge_point_AH = edge_point_A + mid_head_vec
    edge_point_BH = edge_point_B + mid_head_vec

    # return rectangle points
    rect_points = np.zeros((4, 3), dtype=np.float32)
    rect_points[0, :] = edge_point_AH
    rect_points[1, :] = edge_point_A
    rect_points[2, :] = edge_point_B
    rect_points[3, :] = edge_point_BH

    return rect_points
