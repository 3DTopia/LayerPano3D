import depthmap_utils
import projection_icosahedron as proj_ico
import gnomonic_projection as gp
import spherical_coordinates as sc

from EigenSolvers import LinearSolver
from utility import metrics
from utility import depthmap_utils
import matplotlib
# matplotlib.use('TkAgg')


from scipy import ndimage
import scipy.sparse
from scipy.sparse.linalg import spsolve
import scipy.sparse.linalg
import scipy.sparse
import matplotlib.pyplot as plt

import json

import numpy as np

from logger import Logger

log = Logger(__name__)
log.logger.propagate = False
import time


class BlendIt:
    def __init__(self, padding, n_subimages, blending_method):
        # sub-image number
        self.fidelity_weight = 1.0
        self.inflection_point = 10  # point where slope starts to affect the radial blendweights
        self.diagonal_percentage = 48.5  # Percentage of diagonal until weights start decaying in frustum weights
        self.n_subimages = n_subimages
        self.blending_method = blending_method
        self.padding = padding
        self.triangle_coordinates_erp = []  # Pixel coordinates of the triangular tangent face in equirect image
        self.triangle_coordinates_tangent = []
        self.squared_coordinates_erp = []  # Pixel coordinates of the squared (plane) tangent face in equirect image
        self.squared_coordinates_tangent = []
        self.radial_blendweights = None
        self.frustum_blendweights = None
        self.A = None
        self.x_grad_mat = None
        self.y_grad_mat = None
        if blending_method == "all" or blending_method == "poisson":
            # Supported solvers: [SimplicialLLT, SimplicialLDLT, SparseLU, ConjugateGradient,
            #                     LeastSquaresConjugateGradient, BiCGSTAB]
            self.eigen_solver = LinearSolver(LinearSolver.solverType.BiCGSTAB)
        else:
            self.eigen_solver = None

    def blend(self, subimage_dispmap, erp_image_height):
        """Blending the 20 face disparity map to ERP disparity map.

        This function use data in CPU memory, which have been pre-loaded or generated.
        To reduce the time of load data from disk.

        :param subimage_dispmap: A list store the subimage dispartiy data .
        :type subimage_dispmap: list
        :param sub_image_param: The subimage camera parameter.
        :type sub_image_param: dict
        :param erp_image_height: The height of output image.
        :type erp_image_height: float
        """
        tangent_disp_imgs = []
        if isinstance(subimage_dispmap, str):
            for index in range(0, 20):
                tangent_disp_imgs.append(depthmap_utils.read_pfm(subimage_dispmap.format(index))[0])
        elif isinstance(subimage_dispmap, list):
            tangent_disp_imgs = subimage_dispmap
        else:
            log.error("Disparity map type error. {}".format(type(subimage_dispmap)))

        if len(subimage_dispmap) != 20:
            log.error("The blending input subimage size is not 20.")

        # 0) get tangent image information
        erp_image_width = erp_image_height * 2
        erp_size = (erp_image_height, erp_image_width)
        tangent_image_size = subimage_dispmap[0].shape

        # Get erp depth image with as many channels as tangent images along with nearest neighbour (nn) blended image.
        # Also, get erp weight images with the same channels.
        equirect_depth_tensor, nn_blending = self.misc_data(tangent_disp_imgs, erp_size)

        # Normalize
        norm_radial_blendweights = np.divide(self.radial_blendweights,
                                             np.sum(self.radial_blendweights, axis=2)[..., None],
                                             where=(np.sum(self.radial_blendweights, axis=2) > 0)[..., None])

        norm_frustum_blendweights = np.divide(self.frustum_blendweights,
                                              np.sum(self.frustum_blendweights, axis=2)[..., None],
                                              where=(np.sum(self.frustum_blendweights, axis=2) > 0)[..., None])

        radial_blended = np.nansum(norm_radial_blendweights * equirect_depth_tensor, axis=2)
        frustum_blended = np.nansum(norm_frustum_blendweights * equirect_depth_tensor, axis=2)
        mean_blended = np.nanmean(equirect_depth_tensor, axis=2)

        blended_img = dict()
        if self.blending_method == 'poisson':
            blended_img[self.blending_method] = self.gradient_blending(equirect_depth_tensor, self.frustum_blendweights,
                                                                       nn_blending)
        if self.blending_method == 'frustum':
            blended_img[self.blending_method] = frustum_blended

        if self.blending_method == 'radial':
            blended_img[self.blending_method] = radial_blended

        if self.blending_method == 'nn':
            blended_img[self.blending_method] = nn_blending

        if self.blending_method == 'mean':
            blended_img[self.blending_method] = mean_blended

        if self.blending_method == 'all':
            blended_img['poisson'] = self.gradient_blending(equirect_depth_tensor, self.frustum_blendweights,
                                                            nn_blending)
            blended_img['frustum'] = frustum_blended
            blended_img['radial'] = radial_blended
            blended_img['nn'] = nn_blending
            blended_img['mean'] = mean_blended

        return blended_img

    def tangent_images_coordinates(self, erp_image_height, tangent_img_size):
        """
        Based on Mingze's erp2ico_image method in projection_icosahedron.py
        :param tangent_images:
        :param sub_image_param_expression:
        :param erp_size:
        :param tangent_img_size:
        """
        erp_image_width = 2 * erp_image_height
        tangent_image_height, tangent_image_width = tangent_img_size

        if erp_image_width != erp_image_height * 2:
            raise Exception("the ERP image dimession is {}x{}".format(erp_image_height, erp_image_width))

        # stitch all tangnet images to ERP image
        for triangle_index in range(0, 20):
            log.debug("stitch the tangent image {}".format(triangle_index))
            triangle_param = proj_ico.get_icosahedron_parameters(triangle_index, self.padding)

            # 1) get all tangent triangle's available pixels coordinate
            availied_ERP_area = triangle_param["availied_ERP_area"]
            erp_image_col_start, erp_image_row_start = sc.sph2erp(availied_ERP_area[0], availied_ERP_area[2],
                                                                  erp_image_height, sph_modulo=False)
            erp_image_col_stop, erp_image_row_stop = sc.sph2erp(availied_ERP_area[1], availied_ERP_area[3],
                                                                erp_image_height, sph_modulo=False)

            # process the image boundary
            erp_image_col_start = int(erp_image_col_start + 0.5)
            erp_image_col_stop = int(erp_image_col_stop + 0.5)
            erp_image_row_start = int(erp_image_row_start + 0.5)
            erp_image_row_stop = int(erp_image_row_stop + 0.5)

            triangle_x_range = np.linspace(erp_image_col_start, erp_image_col_stop,
                                           erp_image_col_stop - erp_image_col_start, endpoint=False)
            triangle_y_range = np.linspace(erp_image_row_start, erp_image_row_stop,
                                           erp_image_row_stop - erp_image_row_start, endpoint=False)
            triangle_xv, triangle_yv = np.meshgrid(triangle_x_range, triangle_y_range)
            # process the wrap around
            triangle_xv = np.remainder(triangle_xv, erp_image_width)
            triangle_yv = np.clip(triangle_yv, 0, erp_image_height - 1)

            # 2) sample the pixel value from tanget image
            # project spherical coordinate to tangent plane
            spherical_uv = sc.erp2sph([triangle_xv, triangle_yv], erp_image_height=erp_image_height, sph_modulo=False)
            theta_0 = triangle_param["tangent_point"][0]
            phi_0 = triangle_param["tangent_point"][1]
            # Tangent img coordinates normalize to the [0,1]?[-1,1]? tangent plane
            tangent_xv, tangent_yv = gp.gnomonic_projection(spherical_uv[0, :, :], spherical_uv[1, :, :], theta_0,phi_0)

            # the pixels in the tangent triangle
            triangle_points_tangent_nopad = np.array(triangle_param["triangle_points_tangent_nopad"])
            triangle_points_tangent = np.array(triangle_param["triangle_points_tangent"])
            gnomonic_x_min = np.amin(triangle_points_tangent[:, 0], axis=0)
            gnomonic_x_max = np.amax(triangle_points_tangent[:, 0], axis=0)
            gnomonic_y_min = np.amin(triangle_points_tangent[:, 1], axis=0)
            gnomonic_y_max = np.amax(triangle_points_tangent[:, 1], axis=0)

            tangent_gnomonic_range = [gnomonic_x_min, gnomonic_x_max, gnomonic_y_min, gnomonic_y_max]
            pixel_eps = abs(tangent_xv[0, 0] - tangent_xv[0, 1]) / (2 * tangent_image_width)

            square_points_tangent = [[gnomonic_x_min, gnomonic_y_max],
                                     [gnomonic_x_max, gnomonic_y_max],
                                     [gnomonic_x_max, gnomonic_y_min],
                                     [gnomonic_x_min, gnomonic_y_min]]
            inside_tri_pixels_list = gp.inside_polygon_2d(
                np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                triangle_points_tangent_nopad, on_line=True, eps=pixel_eps). \
                reshape(tangent_xv.shape)

            inside_square_pixels_list = gp.inside_polygon_2d(
                np.stack((tangent_xv.flatten(), tangent_yv.flatten()), axis=1),
                square_points_tangent, on_line=True, eps=pixel_eps). \
                reshape(tangent_xv.shape)

            # Tangent coordinates in pixels [subimage_height, subimage_width]
            tangent_sq_xv, tangent_sq_yv = gp.gnomonic2pixel(tangent_xv[inside_square_pixels_list],
                                                             tangent_yv[inside_square_pixels_list],
                                                             0.0, tangent_image_width, tangent_image_height,
                                                             tangent_gnomonic_range)

            tangent_tri_xv, tangent_tri_yv = gp.gnomonic2pixel(tangent_xv[inside_tri_pixels_list],
                                                               tangent_yv[inside_tri_pixels_list],
                                                               0.0, tangent_image_width, tangent_image_height,
                                                               tangent_gnomonic_range)

            self.triangle_coordinates_tangent.append([tangent_tri_xv, tangent_tri_yv])
            self.squared_coordinates_tangent.append([tangent_sq_xv, tangent_sq_yv])
            self.triangle_coordinates_erp.append([triangle_xv[inside_tri_pixels_list], triangle_yv[inside_tri_pixels_list]])
            self.squared_coordinates_erp.append([triangle_xv[inside_square_pixels_list], triangle_yv[inside_square_pixels_list]])

    def erp_blendweights(self, sub_image_param_expression, erp_image_height, tangent_img_size, n_images=20):
        erp_image_width = 2 * erp_image_height
        if erp_image_width != erp_image_height * 2:
            raise Exception("the ERP image dimession is {}x{}".format(erp_image_height, erp_image_width))

        tangent_cam_params = None
        if isinstance(sub_image_param_expression, str):
            tangent_cam_params = json.load(open(sub_image_param_expression.format(0)))
        elif isinstance(sub_image_param_expression, list):
            tangent_cam_params = sub_image_param_expression[0]
        else:
            log.error("Camera parameter type error. {}".format(type(sub_image_param_expression)))

        erp_radial_weights = np.full([erp_image_height, erp_image_width, n_images], 0, np.float64)
        erp_frustum_weights = np.full([erp_image_height, erp_image_width, n_images], 0, np.float64)

        tangent_img_blend_radial_weights = self.get_radial_blendweights(tangent_cam_params, tangent_img_size)

        # mx + b normalization. The slope is -1/(fov-inflection_point)
        # The bias is calculated knowing that in the inflection point we want a weight of 1.
        focal_lengths = np.array([tangent_cam_params['intrinsics']['focal_length_y'],
                                  tangent_cam_params['intrinsics']['focal_length_x']])
        min_f_length = np.argmax(focal_lengths)
        fov = np.degrees(np.arctan(tangent_img_size[min_f_length] * 0.5 / focal_lengths[min_f_length]))
        slope = -1 / (fov - self.inflection_point)
        bias = -self.inflection_point * slope + 1

        # Clip values out of bounds. Weights belong to [0,1]
        tangent_img_blend_radial_weights = np.clip(tangent_img_blend_radial_weights * slope + bias, 0, 1)

        tangent_img_blend_frustum_weights = self.get_frustum_blendweights(tangent_img_size)

        for triangle_index in range(0, n_images):
            tangent_sq_xv, tangent_sq_yv = self.squared_coordinates_tangent[triangle_index]
            erp_sq_xv, erp_sq_yv = self.squared_coordinates_erp[triangle_index]

            erp_face_radial_weights = ndimage.map_coordinates(tangent_img_blend_radial_weights,
                                                              [tangent_sq_yv, tangent_sq_xv],
                                                              order=1, mode='constant', cval=0.)

            erp_face_frustum_weights = ndimage.map_coordinates(tangent_img_blend_frustum_weights,
                                                               [tangent_sq_yv, tangent_sq_xv],
                                                               order=1, mode='constant', cval=0.)

            erp_radial_weights[erp_sq_yv.astype(int), erp_sq_xv.astype(int), triangle_index] = erp_face_radial_weights

            erp_frustum_weights[erp_sq_yv.astype(int), erp_sq_xv.astype(int),
                                triangle_index] = erp_face_frustum_weights

        self.frustum_blendweights = erp_frustum_weights
        self.radial_blendweights = erp_radial_weights

    def misc_data(self, tangent_images, erp_size):
        """
        Based on Mingze's erp2ico_image method in projection_icosahedron.py
        :param tangent_images:
        :param sub_image_param_expression:
        :param erp_size:
        :param tangent_img_size:
        """
        erp_image_height, erp_image_width = erp_size

        erp_depth_tensor = np.full([erp_image_height, erp_image_width, len(tangent_images)], np.nan, np.float64)
        nn_blending = np.zeros(erp_size)

        if erp_image_width != erp_image_height * 2:
            raise Exception("the ERP image dimession is {}".format(np.shape(erp_depth_tensor)))

        # stitch all tangent images to ERP image
        for triangle_index in range(0, 20):
            tangent_tri_xv, tangent_tri_yv = self.triangle_coordinates_tangent[triangle_index]
            tangent_sq_xv, tangent_sq_yv = self.squared_coordinates_tangent[triangle_index]
            erp_tri_xv, erp_tri_yv = self.triangle_coordinates_erp[triangle_index]
            erp_sq_xv, erp_sq_yv = self.squared_coordinates_erp[triangle_index]

            erp_face_image = ndimage.map_coordinates(tangent_images[triangle_index], [tangent_sq_yv, tangent_sq_xv],
                                                     order=1, mode='constant', cval=0.)

            nn_blending[erp_tri_yv.astype(int), erp_tri_xv.astype(int)] = \
                ndimage.map_coordinates(tangent_images[triangle_index], [tangent_tri_yv, tangent_tri_xv],
                                        order=1, mode='constant', cval=0.)

            erp_depth_tensor[erp_sq_yv.astype(int), erp_sq_xv.astype(int),
                             triangle_index] = erp_face_image.astype(np.float64)

        return erp_depth_tensor, nn_blending

    def get_radial_blendweights(self, img_params, size):
        # Weights for each tangent image. Angular distance wrt to the principal point
        height, width = size
        x_list = np.linspace(0, width, width, endpoint=False)
        y_list = np.linspace(0, height, height, endpoint=False)
        grid_x, grid_y = np.meshgrid(x_list, y_list)
        points2d = np.stack((grid_x.ravel(), grid_y.ravel(), np.ones_like(grid_x.ravel())), axis=1).T
        points3d = np.linalg.inv(img_params["intrinsics"]["matrix"]) @ points2d
        points3d = np.divide(points3d, np.linalg.norm(points3d, axis=0))
        points3d = np.moveaxis(points3d.reshape(-1, height, width), 0, -1)
        principal_point = np.array([img_params['intrinsics']['principal_point'][0],
                                    img_params['intrinsics']['principal_point'][1], 1.])
        principal_point_vec = np.linalg.inv(img_params["intrinsics"]["matrix"]) @ principal_point[..., None]
        principal_point_vec = principal_point_vec / np.linalg.norm(principal_point_vec)
        angles = np.degrees(np.arccos(np.clip(np.dot(points3d, principal_point_vec.squeeze()), -1.0, 1.0)))
        return angles

    def get_frustum_blendweights(self, size):
        height, width = size
        weight_matrix = np.zeros((height, width), dtype=float)

        x_list = np.linspace(0, width, width, endpoint=False)
        y_list = np.linspace(0, height, height, endpoint=False)
        grid_x, grid_y = np.meshgrid(x_list, y_list)

        # Distances to y=0, y=height, x=0, x=width lines. They are the 4 lateral planes of the view frustum.
        dist_to_right  = np.abs(grid_x - width)
        dist_to_left   = grid_x
        dist_to_top    = grid_y
        dist_to_bottom = np.abs(grid_y - height)

        # Build pyramid of distances
        total_dist = np.dstack((dist_to_right, dist_to_left, dist_to_top, dist_to_bottom))
        total_dist = np.min(total_dist, axis=2)
        total_dist = (total_dist - np.min(total_dist)) / np.ptp(total_dist)
        peak_coors = np.where(total_dist == 1)
        peak_top_left = np.array([np.min(peak_coors[0]), np.min(peak_coors[1])])
        peak_bottom_right = np.array([np.max(peak_coors[0]), np.max(peak_coors[1])])

        unit_dir = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        top_left = (peak_top_left - 2*self.diagonal_percentage*unit_dir).astype(int)
        bottom_right = (peak_bottom_right + 2*self.diagonal_percentage*unit_dir).astype(int)
        total_dist[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = 0
        total_dist = (total_dist - np.min(total_dist)) / np.ptp(total_dist)
        total_dist[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1] = 1
        return total_dist

    def laplacian_matrix(self, n, m):
        """Generate the Poisson matrix.

        Refer to:
        https://en.wikipedia.org/wiki/Discrete_Poisson_equation

        Note: it's the transpose of the wiki's matrix
        """
        mat_D = scipy.sparse.lil_matrix((m, m))
        mat_D.setdiag(-1, -1)
        mat_D.setdiag(4)
        mat_D.setdiag(-1, 1)

        mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()

        mat_A.setdiag(-1, 1 * m)
        mat_A.setdiag(-1, -1 * m)

        return mat_A

    def concatenate_csc_matrices_by_col(self, matrix1, matrix2):
        new_data = np.concatenate((matrix1.data, matrix2.data))
        new_indices = np.concatenate((matrix1.indices, matrix2.indices))
        new_ind_ptr = matrix2.indptr + len(matrix1.data)
        new_ind_ptr = new_ind_ptr[1:]
        new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

        return scipy.linalg.csc_matrix((new_data, new_indices, new_ind_ptr))

    def concatenate_csr_matrices_by_row(self, blocks):
        data = []
        indices = []
        ind_ptr = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, scipy.sparse.csr_matrix):
                block = block.tocsr()
            data.append(block.data)
            indices.append(block.indices)
            if idx > 0:
                ind_ptr.append((block.indptr + ind_ptr[idx-1][-1])[1:])
            else:
                ind_ptr.append(block.indptr)

        data = np.concatenate(data)
        indices = np.concatenate(indices)
        ind_ptr = np.concatenate(ind_ptr)

        return scipy.sparse.csr_matrix((data, indices, ind_ptr))

    def concatenate_coo_matrices_by_row(self, blocks):
        row_indices = []
        col_indices = []
        data = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, scipy.sparse.coo_matrix):
                block = block.tocoo()
            row_indices.append(block.row + idx*block.shape[0])
            col_indices.append(block.col)
            data.append(block.data)

        row_indices = np.concatenate(row_indices)
        col_indices = np.concatenate(col_indices)
        data = np.concatenate(data)
        return scipy.sparse.coo_matrix((data, (row_indices, col_indices)))

    def gradient_blending(self, equirect_tangent_imgs, equirect_weights, color_blended, eigen_solver=None):
        n_images = equirect_tangent_imgs.shape[2]
        equirect_tangent_imgs[np.isnan(equirect_tangent_imgs)] = 0
        t0 = time.time()

        rows, cols = equirect_tangent_imgs[..., 0].shape

        # matrices = self.get_linear_system_matrices(rows, cols, equirect_weights)

        b = []
        for i in range(0, n_images):
            img = equirect_tangent_imgs[..., i]
            weights = equirect_weights[..., i]

            # grad_x = self.x_grad_mat.dot(img.ravel()).reshape((rows, cols)) * weights
            # grad_y = self.y_grad_mat.dot(img.ravel()).reshape((rows, cols)) * weights
            grad_x = np.diff(img, axis=1, append=img[:, 0, None]) * weights
            grad_y = np.diff(img, axis=0, append=np.zeros_like(img[None, 0])) * weights
            b.append(np.concatenate((grad_x.flatten(), grad_y.flatten())))

        b.append(self.fidelity_weight * color_blended.ravel())
        b = np.concatenate(b)

        if self.eigen_solver is not None:
            x = self.eigen_solver.solve(self.A.transpose().dot(b))
        else:
            x, _ = scipy.sparse.linalg.cg(self.A.transpose().dot(self.A), self.A.transpose().dot(b))
            # x = scipy.sparse.linalg.spsolve(self.A.transpose().dot(self.A), self.A.transpose().dot(b))

        t1 = time.time()
        total = t1 - t0
        print("Blending time = {:3f} (s)".format(total))
        return x.reshape((rows, cols))

    def compute_linear_system_matrices(self, rows, cols, equirect_weights):

        if self.blending_method != "poisson" and self.blending_method != "all":
            return
        # Horizontal forward finite differences
        x_grad_mat = scipy.sparse.coo_matrix((cols, cols))
        x_grad_mat.setdiag(-1)
        x_grad_mat.setdiag(1, 1)
        x_grad_mat = scipy.sparse.lil_matrix(x_grad_mat)
        x_grad_mat[-1, 0] = 1  # Wrap around for edges
        x_grad_mat = scipy.sparse.block_diag([x_grad_mat] * rows).tocsr()

        # Vertical forward finite differences
        y_grad_mat = scipy.sparse.coo_matrix((rows * cols, rows * cols))
        y_grad_mat.setdiag(-1)
        y_grad_mat.setdiag(1, cols)
        y_grad_mat = y_grad_mat.tocsr()

        # Mat A. [x_ffd_1, y_ffd_1, x_ffd_2, y_ffd_2, ... , x_ffd_n, y_ffd_n, fidelity]
        blocks = []
        for i in range(0, self.n_subimages):
            weights = equirect_weights[..., i]

            #   Weight the [x_grad, y_grad] blocks of the A matrix
            blocks.append(x_grad_mat.multiply(weights.ravel()[:, None]))
            blocks.append(y_grad_mat.multiply(weights.ravel()[:, None]))

        blocks.append(self.fidelity_weight * scipy.sparse.eye(blocks[0].shape[1]))
        mat_A = self.concatenate_csr_matrices_by_row(blocks)
        self.A = mat_A
        if self.eigen_solver is not None:
            self.eigen_solver.A = self.A.transpose().dot(self.A)
        # self.x_grad_mat = x_grad_mat
        # self.y_grad_mat = y_grad_mat

