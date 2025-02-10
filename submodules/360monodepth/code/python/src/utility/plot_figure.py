from pathlib import Path

import numpy as np

import projection_icosahedron as pro_ico
import spherical_coordinates as sc
from utility import polygon
from utility import image_io

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False


class PlotFigure():
    """
    This class is used to draw figure for paper.
    """

    def __init__(self):
        self.icosahedron_radius = 1.0
        self.output_dir = None

    def draw_ico_tangent_planes_texture_stitch(self, root_dir, subimage_filename_list, texture_filename):
        """ Stitch all 20 subimage rgb image (W * H) to a single long image ( 8W * H).
        
        :return:
        :rtype: numpy
        """
        subimage_data_list = []
        subimage_width = -1
        subimage_height = -1
        subimage_channel = None
        for subimage_filename in subimage_filename_list:
            subimage_data = image_io.image_read(root_dir + subimage_filename)
            subimage_data_list.append(subimage_data)

            subimage_width = subimage_data.shape[1]
            subimage_height = subimage_data.shape[0]
            subimage_channel = subimage_data.shape[2]

            # if texture_height < 0:
            #     texture_height = subimage_height
            # elif texture_height > 0 and texture_height != subimage_height:
            #     log.error("the subimage size is different!")

        if subimage_channel != 3:
            log.warn("Remove the subimage alpha channel.")

        texture_height = subimage_height * 4
        texture_width = subimage_width * 5

        texture_data = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
        texture_subimage_row_start = 0

        # for idx in range(len(subimage_filename_list)):
        #     texture_subimage_end = texture_subimage_start + subimage_data_list[idx].shape[1]

        #     texture_subimage_start = texture_subimage_end
        for row_idx in range(4):
            texture_subimage_row_end = texture_subimage_row_start + subimage_height
            texture_subimage_col_start = 0
            for col_idx in range(5):
                texture_subimage_col_end = texture_subimage_col_start + subimage_width
                texture_data[texture_subimage_row_start:texture_subimage_row_end, texture_subimage_col_start: texture_subimage_col_end, :] \
                    = subimage_data_list[row_idx * 5 + col_idx][:, :, 0:3]

                texture_subimage_col_start = texture_subimage_col_end
            texture_subimage_row_start = texture_subimage_row_end

        # return texture_data
        image_io.image_save(texture_data, root_dir + texture_filename)
        # im.save('uncompressed.tga', compression=None)

    def draw_ico_tangent_planes(self, radius, padding_size, obj_file_path, subimage_shift_ratio=0.0,         obj_file_texture_enable=False, texture_image_filename="ico_rgb_image.png"):
        """ Draw the ico's 3D tangent plan.

        :param radius: the radius of icosahedron.
        :type radius: float
        :param radius: the tangent image padding size
        :type radius: float
        :param obj_file_path: the obj file output path.
        :type obj_file_path: str
        """
        # 1) the panes center 3D center point
        if obj_file_texture_enable:
            mtl_file_path_path = Path(obj_file_path)
            mtl_file_path_path = mtl_file_path_path.with_suffix('.mtl')
            obj_file_head = "mtllib {}\n".format(str(mtl_file_path_path))
            obj_file_head += "usemtl material_0\n"
            obj_file_texture_coor_str = ""
        else:
            obj_file_head = ""

        obj_file_vertex_str = ""
        # the texture coordinate
        obj_file_face_str = ""
        obj_file_line_str = ""

        for tangent_image_idx in range(0, 20, 1):
            tangent_image_param = \
                pro_ico.get_icosahedron_parameters(tangent_image_idx, padding_size)

            triangle_points_sph = tangent_image_param["triangle_points_sph"]
            triangle_points_sph = np.array(triangle_points_sph, dtype=np.float32)

            triangle_points_sph_sort_idx = np.argsort(triangle_points_sph[:, 0])
            if (triangle_points_sph_sort_idx != np.array((0, 1, 2))).any():
                triangle_points_sph_sort_idx = np.tile(triangle_points_sph_sort_idx, 2).reshape((2, 3))
                triangle_points_sph = np.take_along_axis(triangle_points_sph, triangle_points_sph_sort_idx.T, axis=0)

            triangle_points_3d = sc.sph2car(triangle_points_sph[:, 0], triangle_points_sph[:, 1], radius=radius)
            # find the head vertex
            triangle_points_3d_head = None
            triangle_points_3d_edge = np.zeros((2, 3), dtype=np.float32)
            if triangle_points_sph[0, 1] == triangle_points_sph[1, 1]:
                triangle_points_3d_head = triangle_points_3d[:, 2]
                triangle_points_3d_edge[0, :] = triangle_points_3d[:, 0]
                triangle_points_3d_edge[1, :] = triangle_points_3d[:, 1]
            elif triangle_points_sph[0, 1] == triangle_points_sph[2, 1]:
                triangle_points_3d_head = triangle_points_3d[:, 1]
                triangle_points_3d_edge[0, :] = triangle_points_3d[:, 0]
                triangle_points_3d_edge[1, :] = triangle_points_3d[:, 2]
            else:
                triangle_points_3d_head = triangle_points_3d[:, 0]
                triangle_points_3d_edge[0, :] = triangle_points_3d[:, 1]
                triangle_points_3d_edge[1, :] = triangle_points_3d[:, 2]

            # 2) Output the vertices, texture information
            # the tangent plane rectangle tangent images
            if subimage_shift_ratio != 0.0:
                # offset the tangent images
                triangle_normal_norm_sph = tangent_image_param["tangent_point"]
                triangle_normal_norm_3d = sc.sph2car(triangle_normal_norm_sph[0], triangle_normal_norm_sph[1], radius=radius)
                subimage_shift = triangle_normal_norm_3d * subimage_shift_ratio
                triangle_points_3d_edge += subimage_shift
                triangle_points_3d_head += subimage_shift

            # 2-0) 3D points
            rectangle_points_3d = polygon.triangle_bounding_rectangle_3D(triangle_points_3d_head, triangle_points_3d_edge)
            # write vertex
            for vec_idx in range(4):
                vec_val = rectangle_points_3d[vec_idx]
                obj_file_vertex_str += "v {} {} {}\n".format(vec_val[0], vec_val[1], vec_val[2])

            # 2-1) add texture
            # write texture name and image file
            if obj_file_texture_enable:
                # texture file information
                # obj_file_face_str += "usemtl material_0\n"

                # texture offset
                texture_x_offset_start = 1.0 / 5.0 * (tangent_image_idx % 5)
                texture_x_offset_end = 1.0 / 5.0 + texture_x_offset_start
                texture_y_offset_start = 1.0 - 1.0 / 4.0 * int(tangent_image_idx / 5)
                texture_y_offset_end = texture_y_offset_start - 1.0 / 4.0
                obj_file_texture_coor_str += "vt {} {}\n".format(texture_x_offset_start, texture_y_offset_start)
                obj_file_texture_coor_str += "vt {} {}\n".format(texture_x_offset_start, texture_y_offset_end)
                obj_file_texture_coor_str += "vt {} {}\n".format(texture_x_offset_end, texture_y_offset_end)
                obj_file_texture_coor_str += "vt {} {}\n".format(texture_x_offset_end, texture_y_offset_start)

            # 2-2) write face
            # the head vertex alway at the top, make a face outside
            head_vertex_up = triangle_points_3d_head[1] < triangle_points_3d_edge[0, 1]
            # if head_vertex_up:
            #     triangle_points_3d_edge[[0, 1], :] = triangle_points_3d_edge[[1, 0], :]
            idx_offset = tangent_image_idx * 4
            if obj_file_texture_enable:
                if head_vertex_up:
                    print("{}:up".format(tangent_image_idx))
                    obj_file_face_str += "f {}/{} {}/{} {}/{}\n".format(idx_offset + 1, 1 + idx_offset,
                                                                        idx_offset + 2, 2 + idx_offset,
                                                                        idx_offset + 3, 3 + idx_offset)
                    obj_file_face_str += "f {}/{} {}/{} {}/{}\n".format(idx_offset + 1, 1 + idx_offset,
                                                                        idx_offset + 3, 3 + idx_offset,
                                                                        idx_offset + 4, 4 + idx_offset)
                else:
                    print("{}:down".format(tangent_image_idx))
                    obj_file_face_str += "f {}/{} {}/{} {}/{}\n".format(idx_offset + 3, 1 + idx_offset,
                                                                        idx_offset + 4, 2 + idx_offset,
                                                                        idx_offset + 1, 3 + idx_offset)
                    obj_file_face_str += "f {}/{} {}/{} {}/{}\n".format(idx_offset + 3, 1 + idx_offset,
                                                                        idx_offset + 1, 3 + idx_offset,
                                                                        idx_offset + 2, 4 + idx_offset)
            else:
                obj_file_face_str += "f {} {} {}\n".format(idx_offset + 1, idx_offset + 2, idx_offset + 3)
                obj_file_face_str += "f {} {} {}\n".format(idx_offset + 1, idx_offset + 3, idx_offset + 4)

            obj_file_line_str += "f {} {} {} {}\n".format(idx_offset + 1, idx_offset + 2, idx_offset + 3, idx_offset + 4)

        # 3) write obj file
        # 3-0) tangent images obj file
        # write tangent image obj file
        log.info("output the mesh to {}".format(obj_file_path))
        with open(obj_file_path, "w") as obj_file:
            obj_file.write(obj_file_head)
            obj_file.write(obj_file_vertex_str)
            if obj_file_texture_enable:
                obj_file.write(obj_file_texture_coor_str)

            obj_file.write(obj_file_face_str)

        # write tangent image mtl files
        if obj_file_texture_enable:
            # output mtl
            mtl_file_str = ""
            mtl_file_str += "newmtl material_0\n"
            mtl_file_str += "Ka 0.200000 0.200000 0.200000\n"
            mtl_file_str += "Kd 1.000000 1.000000 1.000000\n"
            mtl_file_str += "Ks 1.000000 1.000000 1.000000\n"
            mtl_file_str += "map_Kd {}\n\n".format(texture_image_filename)
            with open(str(mtl_file_path_path), "w") as mtl_file:
                mtl_file.write(mtl_file_str)

        # 3-1) write framewire obj file
        # write line obj
        obj_framewire_obj_file = obj_file_path + "_frameware.obj"
        log.info("output the framewire to {}".format(obj_file_path))
        with open(obj_framewire_obj_file, "w") as obj_file:
            obj_file.write(obj_file_vertex_str)
            obj_file.write(obj_file_line_str)
