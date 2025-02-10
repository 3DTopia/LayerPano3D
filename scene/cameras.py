#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np

import torch
from torch import nn
from PIL import Image

from utils.graphics import getWorld2View2, getProjectionMatrix, getProjectionMatrix2, fov2focal
from utils.loss import image2canny
from utils.general import PILtoTorch



class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.canny_mask = image2canny(self.original_image.permute(1,2,0), 50, 150, isEdge1=False).detach().to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Camera2(nn.Module):
    def __init__(self, mini_cam, image, data_device='cuda'):

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
            
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        
        self.image_width = mini_cam.image_width
        self.image_height = mini_cam.image_height
        
        self.FoVx = mini_cam.FoVx
        self.FoVy = mini_cam.FoVy

        
        self.world_view_transform = mini_cam.world_view_transform
        self.projection_matrix = mini_cam.projection_matrix
        self.full_proj_transform = mini_cam.full_proj_transform
        self.camera_center = mini_cam.camera_center

        self.zfar = mini_cam.zfar
        self.znear = mini_cam.znear

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        


class MiniCam2:
    def __init__(self, pose, width, height, fovx, fovy):
        
        self.image_width = width
        self.image_height = height
        
        self.FoVx = fovx
        self.FoVy = fovy

        # focalx = fov2focal(fovx, width)
        # focaly = fov2focal(fovy, height)
        # print('INFO: focalx, width, focaly, height', focalx, width, focaly, height) 

        w2c = np.linalg.inv(pose)
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1


        self.znear, self.zfar = 0.01, 100

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix2(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )

        self.world_view_transform = self.world_view_transform.to(torch.float32)
        self.projection_matrix =  self.projection_matrix.to(torch.float32)


        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(pose[:3, 3]).cuda().to(torch.float32)

class MiniCam_GS:
    def __init__(self, pose, width, height, fovx, fovy):
        
        self.image_width = width
        self.image_height = height
        
        self.FoVx = fovx
        self.FoVy = fovy

        # focalx = fov2focal(fovx, width)
        # focaly = fov2focal(fovy, height)
        # print('INFO: focalx, width, focaly, height', focalx, width, focaly, height) 

        w2c = np.linalg.inv(pose)
        # w2c[1:3, :3] *= -1
        # w2c[:3, 3] *= -1


        self.znear, self.zfar = 0.01, 100

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda()
        self.projection_matrix = (
            getProjectionMatrix2(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )

        self.world_view_transform = self.world_view_transform.to(torch.float32)
        self.projection_matrix =  self.projection_matrix.to(torch.float32)


        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(pose[:3, 3]).cuda().to(torch.float32)


class MiniCam_4d:
    def __init__(self, pose, width, height, fovx, fovy, time):
        
        self.image_width = width
        self.image_height = height
        
        
        self.FoVx = fovx
        self.FoVy = fovy

        c2w=pose_preprocess(pose)
        
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        self.znear, self.zfar = 0.01, 100
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        self.time = time

class MiniCam2_4d:
    def __init__(self, pose, width, height, fovx, fovy, time, image, gt_alpha_mask=None):
        
        self.image_width = width
        self.image_height = height
        
        
        image = PILtoTorch(image)
            
        self.original_image = image.clip(0.0, 1.0)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
            # .to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
                                                #   , device=self.data_device)
        
        
        self.FoVx = fovx
        self.FoVy = fovy

        c2w=pose_preprocess(pose)
        
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        self.znear, self.zfar = 0.01, 100
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

        self.time = time


def pose_preprocess(pose):
    yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
    
    ### Transform world to pixel
    Rw2i = pose[:3,:3]
    Tw2i = pose[:3,3:4]

    # Transfrom cam2 to world + change sign of yz axis
    Ri2w = np.matmul(yz_reverse, Rw2i).T
    Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
    Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
    Pc2w = np.concatenate((Pc2w, np.array([0,0,0,1]).reshape((1,4))), axis=0)
    return Pc2w




# class MiniCam2:
#     def __init__(self, pose, width, height, fovx, fovy):
        
#         self.image_width = width
#         self.image_height = height
        
#         self.FoVx = fovx
#         self.FoVy = fovy

#         c2w=pose_preprocess(pose)
        
#         c2w[:3, 1:3] *= -1
#         # get the world-to-camera transform and set R, T
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]
        
#         self.znear, self.zfar = 0.01, 100
#         self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).cuda()
#         self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
#         self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
#         view_inv = torch.inverse(self.world_view_transform)
#         self.camera_center = view_inv[3][:3]