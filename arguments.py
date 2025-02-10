###
# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
# All rights reserved.
###
import numpy as np
import math
from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelHiddenParams:
    def __init__(self):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0001
        self.time_smoothness_weight = 0.01
        self.l1_time_planes = 0.0001
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]
                            }
        self.multires = [1, 2, 4, 8]
        self.no_grid=False
        self.no_ds=False
        self.no_dr=False
        self.no_do=True

        
        # super().__init__(parser, "ModelHiddenParams")



class GSParams: 
    def __init__(self):
        self.sh_degree = 3
        self.images = "images"
        self.resolution = -1
        self.white_background = True
        self.data_device = "cuda"
        self.eval = False
        self.use_depth = False

        self.iterations = 8990#3_000
        
        self.iterations_4d = 30_000
        
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 8990#3_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        
        self.densification_interval = 100  ###
        self.opacity_reset_interval = 10000 ###
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002

        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

        self.lambda_dssim = 0.2
        self.lambda_lpips = 0
        
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002

        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005

        self.pruning_from_iter = 500
        self.pruning_interval = 100


# class GSParams: 
#     def __init__(self):
#         self.sh_degree = 3
#         self.images = "images"
#         self.resolution = -1
#         self.white_background = True
#         self.data_device = "cuda"
#         self.eval = False
#         self.use_depth = False

#         self.iterations = 8990#3_000
        
#         self.iterations_4d = 30_000
        
#         self.position_lr_init = 0.00016
#         self.position_lr_final = 0.0000016
#         self.position_lr_delay_mult = 0.01
#         self.position_lr_max_steps = 8990#3_000
#         self.feature_lr = 0.0025
#         self.opacity_lr = 0.05
#         self.scaling_lr = 0.005
#         self.rotation_lr = 0.001
#         self.percent_dense = 0.01
        
#         self.densification_interval = 100
#         self.opacity_reset_interval = 10000
#         self.densify_from_iter = 500
#         self.densify_until_iter = 15_000
#         self.densify_grad_threshold = 0.0002

#         self.convert_SHs_python = False
#         self.compute_cov3D_python = False
#         self.debug = False

#         self.lambda_dssim = 0.2
#         self.lambda_lpips = 0
        
#         self.densify_grad_threshold_coarse = 0.0002
#         self.densify_grad_threshold_fine_init = 0.0002
#         self.densify_grad_threshold_after = 0.0002

#         self.opacity_threshold_coarse = 0.005
#         self.opacity_threshold_fine_init = 0.005
#         self.opacity_threshold_fine_after = 0.005

#         self.pruning_from_iter = 500
#         self.pruning_interval = 100



def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


class CameraParams:
    def __init__(self, H: int = 512, W: int = 512, fov=90):
        self.H = H
        self.W = W
        
        self.fovx = math.radians(fov)
        self.fovy = self.H * self.fovx / self.W

        # self.fovy = math.radians(fov)        
        # self.fovx = self.W * self.fovy / self.H        

        self.fov = (self.fovx, self.fovy)
        self.fov_deg = fov
        
        
        self.fx = fov2focal(self.fovx, self.W) 
        self.fy = fov2focal(self.fovy, self.H)
        
        # print("focal",self.fx, self.fy)
        
        self.K = np.array([
            [self.fx, 0., self.W/2],
            [0., self.fy, self.H/2],
            [0.,      0.,       1.],
        ]).astype(np.float32)

        # self.H = H
        # self.W = W
        # self.focal = (5.8269e+02, 5.8269e+02)
        # self.fov = (2*np.arctan(self.W / (2*self.focal[0])), 2*np.arctan(self.H / (2*self.focal[1])))
        # self.K = np.array([
        #     [self.focal[0], 0., self.W/2],
        #     [0., self.focal[1], self.H/2],
        #     [0.,            0.,       1.],
        # ]).astype(np.float32)

