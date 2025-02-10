import spherical_coordinates as sc

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import struct

from logger import Logger
log = Logger(__name__)
log.logger.propagate = False

"""
Point cloud utility.
"""

def depthmap2pointcloud_erp(depth_map, rgb_image, output_ply_file_path):
    """ Convert the ERP depth map and rgb_image to 3D colored point cloud.

    :param depth_map: The ERP depth map, shape is [height, width]
    :type depth_map: numpy
    :param rgb_image: The ERP rgb image, shape is [height, widht, 3]
    :type rgb_image: numpy
    :param output_ply_file_path: 3D point cloud output path.
    :type output_ply_file_path: str
    """
    # spherical coordinate
    pixel_x, pixel_y = np.meshgrid(range(depth_map.shape[1]), range(rgb_image.shape[0]))
    theta, phi = sc.erp2sph([pixel_x, pixel_y])

    # spherical coordinate to point cloud
    x = (depth_map * np.cos(phi) * np.sin(theta)).flatten()
    y = - (depth_map * np.sin(phi)).flatten()
    z = (depth_map * np.cos(phi) * np.cos(theta)).flatten()

    r = rgb_image[pixel_y, pixel_x, 0].flatten()
    g = rgb_image[pixel_y, pixel_x, 1].flatten()
    b = rgb_image[pixel_y, pixel_x, 2].flatten()

    # print("depthmap2pointcloud_erp", pixel_x.shape, pixel_y.shape, theta.shape, phi.shape, x.shape, r.shape,r,g,b)
    # depthmap2pointcloud_erp (1024, 2048) (1024, 2048) (1024, 2048) (1024, 2048) (2097152,) (2097152,) [ 42  32  34 ... 216 217 217] [161 163 161 ... 220 221 222] [218 223 222 ... 205 204 203]
    
    # return np.stack([x, y, z], axis=1)
    point_cloud_data = np.stack([x, y, z, r, g, b], axis=1)

    # 2)  Output point cloud to obj file.
    # # convert to vertices list
    # obj_text = []
    # for line in point_cloud_data.T:
    #     v_list = line.tolist()
    #     obj_text.append("v {} {} {}".format(v_list[0], v_list[1], v_list[2]))

    # # output to target obj file
    # # print("generate obj file {}".format(obj_file_path))
    # output_obj_handle = open(output_obj_file_path, 'w')
    # obj_points_str = '\n'.join(obj_text)
    # output_obj_handle.write(obj_points_str)
    # output_obj_handle.close()

    # output color point cloud to PLY

    # Write header of .ply file
    fid = open(output_ply_file_path, 'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n' % point_cloud_data.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(point_cloud_data.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", point_cloud_data[i, 0], point_cloud_data[i, 1], point_cloud_data[i, 2],
                                        bytes(point_cloud_data[i, 3].astype(np.uint8).data),
                                        bytes(point_cloud_data[i, 4].astype(np.uint8).data),
                                        bytes(point_cloud_data[i, 5].astype(np.uint8).data))))

    fid.close()


def depthmap2pointclouds_perspective(depth_map, rgb_image, cam_int_param, output_path, rgb_image_path=None):
    """Convert the depth map to 3D mesh and export to file.

    The input numpy array is [height, width, x].

    :param depth_map: depth map
    :type depth_map: numpy
    :param rgb_image: rgb image data
    :type rgb_image: numpy
    :param cam_int_param: camera 3x3 calibration matrix, [[fx, 0, cx], [0,fy,cy], [0,0,1]]
    :type cam_int_param: numpy
    :param output_path: the absolute path of output mesh, support ply and obj.
    :type output_path: str
    :param rgb_image_path: the rgb image relative path, used by obj's mtl file.
    :type rgb_image_path: str
    """
    # 1) check the function arguments
    _, output_path_ext = os.path.splitext(output_path)

    if not output_path_ext == ".obj" and not output_path_ext == ".ply":
        log.error("Current do not support {}  format".format(output_path_ext[1:]))

    # 2) convert the depth map to 3d points
    image_height = depth_map.shape[0]
    image_width = depth_map.shape[1]

    x_list = np.linspace(0, image_width, image_width, endpoint=False)
    y_list = np.linspace(0, image_height, image_height, endpoint=False)
    grid_x, grid_y = np.meshgrid(x_list, y_list)
    gird_z = np.ones(grid_x.shape, float)
    points_2d_pixel = np.stack((grid_x.ravel(), grid_y.ravel(), gird_z.ravel()), axis=1)
    points_2d_pixel = np.multiply(points_2d_pixel.T, depth_map.ravel())
    points_3d_pixel = np.linalg.inv(cam_int_param) @ points_2d_pixel
    points_3d_pixel = points_3d_pixel.T.reshape((depth_map.shape[:2] + (3,)))

    # 3) output to file
    log.debug("save the mesh to {}".format(output_path_ext))

    if output_path_ext == ".obj":
        if os.path.exists(output_path):
            log.warn("{} exist, overwrite it.".format(output_path))

        output_mtl_path = None
        if rgb_image is None or rgb_image_path is None:
            rgb_image_path = None
            output_mtl_path = None
            log.debug("Do not specify texture for the obj file.")
        else:
            output_mtl_path = os.path.splitext(output_path)[0] + ".mtl"
            if os.path.exists(output_mtl_path):
                log.warn("{} exist, overwrite it.".format(output_mtl_path))

        create_obj(depth_map, points_3d_pixel, output_path, output_mtl_path, texture_filepath=rgb_image_path)
    elif output_path_ext == ".ply":
        log.critical("do not implement!")
        # create_ply()


def pointcloud_tang2world(point_cloud_data, tangent_point):
    """ Rotation tangent point cloud to world coordinate system.
    Tranfrom the point cloud from tangent space to world space.

    :param point_cloud_data: The point cloud array [3, points_number]
    :type point_cloud_data: numpy 
    :param tangent_point:  the tangent point rotation [theta,phi] in radiant.
    :type tangent_point: list
    """
    assert len(point_cloud_data.shape) == 2
    assert point_cloud_data.shape[0] == 3

    rotation_matrix = R.from_euler("xyz", [tangent_point[1], tangent_point[0], 0], degrees=False).as_dcm()
    xyz_rotated = np.dot(rotation_matrix, point_cloud_data)
    return xyz_rotated


def create_obj(depthmap, point3d, obj_filepath, mtl_filepath=None, mat_name="material0", texture_filepath=None):
    """This method does the same as :func:`depthmap2mesh`
    """
    use_material = False
    if mtl_filepath is not None:
        use_material = True

    # create mtl file
    if use_material:
        with open(mtl_filepath, "w") as f:
            f.write("newmtl " + mat_name + "\n")
            f.write("Ns 10.0000\n")
            f.write("d 1.0000\n")
            f.write("Tr 0.0000\n")
            f.write("illum 2\n")
            f.write("Ka 1.000 1.000 1.000\n")
            f.write("Kd 1.000 1.000 1.000\n")
            f.write("Ks 0.000 0.000 0.000\n")
            f.write("map_Ka " + texture_filepath + "\n")
            f.write("map_Kd " + texture_filepath + "\n")

    # create obj file
    width = depthmap.shape[1]
    hight = depthmap.shape[0]

    with open(obj_filepath, "w") as file:
        # output
        if use_material:
            file.write("mtllib " + mtl_filepath + "\n")
            file.write("usemtl " + mat_name + "\n")

        # the triangle's vertex index
        pixel_vertex_index = np.zeros((width, hight), int)  # pixels' vertex index number
        vid = 1  # vertex index

        # output vertex
        for u in range(0, width):
            for v in range(hight-1, -1, -1):
                # vertex index
                pixel_vertex_index[v, u] = vid
                if depthmap[v, u] == 0.0:
                    pixel_vertex_index[u, v] = 0
                vid += 1

                # output 3d vertex
                x = point3d[v, u, 0]
                y = point3d[v, u, 1]
                z = point3d[v, u, 2]
                file.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        # output texture location
        for u in range(0, width):
            for v in range(0, hight):
                file.write("vt " + str(u/width) + " " + str(v/hight) + "\n")

        # output face index
        for u in range(0, width-1):
            for v in range(0, hight-1):
                v1 = pixel_vertex_index[u, v]
                v2 = pixel_vertex_index[u+1, v]
                v3 = pixel_vertex_index[u, v+1]
                v4 = pixel_vertex_index[u+1, v+1]

                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue

                file.write("f " + str(v1)+"/"+str(v1) + " " + str(v2)+"/"+str(v2) + " " + str(v3)+"/"+str(v3) + "\n")
                file.write("f " + str(v3)+"/"+str(v3) + " " + str(v2)+"/"+str(v2) + " " + str(v4)+"/"+str(v4) + "\n")
