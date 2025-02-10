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
import os

import numpy as np
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from simple_knn._C import distCUDA2
from utils.general import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.system import mkdir_p
from utils.sh import RGB2SH
from utils.graphics import BasicPointCloud
from utils.general import strip_symmetric, build_scaling_rotation
from tqdm import tqdm
from scipy.spatial import cKDTree
from datetime import datetime

class LayerGaussian:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, outlier_thresh = 4):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._xyz_static = torch.empty(0)
        self._features_dc_static = torch.empty(0)
        self._features_rest_static = torch.empty(0)
        self._scaling_static = torch.empty(0)
        self._rotation_static = torch.empty(0)
        self._opacity_static = torch.empty(0)
        self._xyz_surface = torch.empty(0)
        self._features_dc_surface = torch.empty(0)
        self._features_rest_surface = torch.empty(0)
        self._scaling_surface = torch.empty(0)
        self._rotation_surface = torch.empty(0)
        self._opacity_surface = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.outlier_thresh = outlier_thresh
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def capture_total(self):
        f_dc = torch.cat((self._features_dc_static, self._features_dc),dim=0)
        f_rest = torch.cat((self._features_rest_static, self._features_rest),dim=0)
        opacities = torch.cat([self._opacity_static, self._opacity], dim=0)
        _scaling = torch.cat([self._scaling_static, self._scaling], dim=0)
        _rotation = torch.cat([self._rotation_static, self._rotation], dim=0)
        xyz_gradient_accum_static =  torch.zeros((self._xyz_static.shape[0], 1), device="cuda")
        xyz_gradient_accum = torch.cat([xyz_gradient_accum_static, self.xyz_gradient_accum], dim=0)
        max_radii2D_static = torch.zeros((self._xyz_static.shape[0]), device="cuda")
        max_radii2D = torch.cat([max_radii2D_static, self.max_radii2D], dim=0)
        denom_static = torch.zeros((self._xyz_static.shape[0], 1), device="cuda")
        denom = torch.cat([denom_static, self.denom], dim=0)

        return (
            self.active_sh_degree,
            self.get_xyz_total,
            f_dc,
            f_rest,
            _scaling,
            _rotation,
            opacities,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.identity
        )

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_total(self):
        return self.scaling_activation(torch.cat([self._scaling_static, self._scaling],dim=0))
    
    @property
    def get_scaling_render(self):
        return self.scaling_activation(torch.cat([self._scaling_surface, self._scaling],dim=0))

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_rotation_total(self):
        return self.rotation_activation(torch.cat([self._rotation_static, self._rotation],dim=0))
    
    @property
    def get_rotation_render(self):
        return self.rotation_activation(torch.cat([self._rotation_surface, self._rotation],dim=0))
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_xyz_total(self):
        return torch.cat((self._xyz_static, self._xyz),dim=0)

    
    @property
    def get_xyz_render(self):
        return torch.cat((self._xyz_surface, self._xyz),dim=0)
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_total(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features = torch.cat((features_dc, features_rest), dim=1)
        features_dc_static = self._features_dc_static
        features_rest_static = self._features_rest_static
        features_static = torch.cat((features_dc_static, features_rest_static), dim=1)
        return torch.cat((features_static, features), dim=0)

    @property
    def get_features_render(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features = torch.cat((features_dc, features_rest), dim=1)
        features_dc_surface = self._features_dc_surface
        features_rest_surface = self._features_rest_surface
        # print('debug: get_features_render',features_dc_surface.shape, features_rest_surface.shape)
        features_surface = torch.cat((features_dc_surface, features_rest_surface), dim=1)
        return torch.cat((features_surface, features), dim=0)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_opacity_total(self):
        return self.opacity_activation(torch.cat([self._opacity_static, self._opacity], dim=0))

    @property
    def get_opacity_render(self):
        return self.opacity_activation(torch.cat([self._opacity_surface, self._opacity], dim=0))
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_covariance_total(self, scaling_modifier = 1):
        rotation_total = torch.cat([self._rotation_static, self._rotation], dim=0)
        return self.covariance_activation(self.get_scaling_total, scaling_modifier, rotation_total)

    def get_covariance_render(self, scaling_modifier = 1):
        rotation_total = torch.cat([self._rotation_surface, self._rotation], dim=0)
        return self.covariance_activation(self.get_scaling_render, scaling_modifier, rotation_total)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def update_identity_mask(self):
        self.identity_mask = (self.identity > 0)
        edge_mask = (self.identity == 3)
        self.edge_mask = edge_mask[len(self._xyz_surface):]

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_from_pcd_and_masks(self, pcd : BasicPointCloud, masks : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        identity = torch.tensor(np.asarray(masks.colors)[:,0]).float().cuda()
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        min_dist = distCUDA2(fused_point_cloud)
        # def detect_outliers_zscore(data):
        #     mean = torch.mean(data)
        #     std = torch.std(data)
        #     z_scores = [(x - mean) / std for x in data]
        #     mask = torch.zeros_like(data).bool()
        #     for i, z in tqdm(enumerate(z_scores)):
        #         if abs(z) > 3:
        #             mask[i] = True
        #     # outliers = [data[i]  if abs(z) > 3]
        #     return mask
        dist_mask = min_dist > 0.0001
        print('min_dist > 0.001 removed {} points'.format(torch.sum(dist_mask)))
        # dist_mask = detect_outliers_zscore(min_dist)
        # print('detect outliers zscore removed {} points'.format(torch.sum(dist_mask)))
        bagel_temp = identity > 0 
        # bagel_temp = identity == 0
        bagel_debug = (identity < 1) & (identity > 0) 
        
        dist_mask = torch.logical_and(dist_mask, bagel_debug)
        print('Removed {} Bagel points'.format(dist_mask.sum()))
        identity[bagel_debug] = 0
        identity_mask = identity > 0


        fused_point_cloud = fused_point_cloud[identity_mask]
        identity_temp = identity[identity_mask]
        # bagel_temp = identity_temp < 1
        outlier_mask = self.get_outlier_filter(fused_point_cloud.detach().cpu().numpy(), bagel_temp.detach().cpu().numpy(),thresh=self.outlier_thresh)
        
        # outlier_mask[:] = True # budapest layer 2

        fused_point_cloud = fused_point_cloud[outlier_mask]

        # identity_temp2 = identity[identity_mask][outlier_mask]
        # bagel_temp2 = identity_temp2 < 1
        # outlier_mask2 = self.get_outlier_filter2(fused_point_cloud.detach().cpu().numpy(), bagel_temp2.detach().cpu().numpy(),thresh=20)
        # outlier_mask2[:] = True
        
        # outlier_indices = np.where(outlier_mask)[0]
        # outlier_mask[outlier_indices] = outlier_mask2
        outlier_mask = torch.tensor(outlier_mask).cuda()
        # outlier_mask2 = torch.tensor(outlier_mask2).cuda()
        # fused_point_cloud = fused_point_cloud[outlier_mask2]
        

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors), device='cuda')[identity_mask][outlier_mask].float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of new pcd points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        # print('dist2 and pcd points shape', dist2.shape, fused_point_cloud.shape)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        _xyz = fused_point_cloud.requires_grad_(True)
        _features_dc = features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True)
        _features_rest = features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True)
        _scaling = scales.requires_grad_(True)
        _rotation = rots.requires_grad_(True)
        _opacity = opacities.requires_grad_(True)

        bagel_mask = torch.logical_and((identity > 0),(identity < 1))
        identity[bagel_mask] = 3.0

        # xyz_part = np.asarray(pcd.points)[identity_mask.detach().cpu().numpy()]
        # colors_part = np.asarray(pcd.colors)[identity_mask.detach().cpu().numpy()]
        timestamp = datetime.now().strftime("%m%d_%H%M%S")

        # save_pcd(f'debug/pcd_part_0_{timestamp}.ply', xyz_part, colors_part)
        

        xyz_part = np.asarray(pcd.points)[identity_mask.detach().cpu().numpy()][outlier_mask.detach().cpu().numpy()]
        colors_part = np.asarray(pcd.colors)[identity_mask.detach().cpu().numpy()][outlier_mask.detach().cpu().numpy()]

        # xyz_part = np.asarray(pcd.points)[identity_mask.detach().cpu().numpy()][outlier_mask.detach().cpu().numpy()][outlier_mask2.detach().cpu().numpy()]
        # colors_part = np.asarray(pcd.colors)[identity_mask.detach().cpu().numpy()][outlier_mask.detach().cpu().numpy()][outlier_mask2.detach().cpu().numpy()]
        # save_pcd(f'debug/pcd_part_2_{timestamp}.ply', xyz_part, colors_part)

        # 1.0 nornal pcd points
        # 2.0 active inherited gaussian points
        # 3.0 edge points

        
        return identity[identity_mask][outlier_mask], _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, identity_mask.detach().cpu().numpy(), outlier_mask.detach().cpu().numpy()      
    
    def load_ply_and_create_pcd(self, plydata, pcd, masks, spatial_lr_scale):
        # plydata = {'xyz':xyz, 'normals':normals, 'f_dc':f_dc, 'f_rest':f_rest, 'opacities':opacities, 'scale':scale, 'rotation':rotation}
        if plydata is not None:
            xyz, opacities, features_dc, features_extra, scales, rots = plydata['xyz'], plydata['opacities'], plydata['f_dc'], plydata['f_rest'], plydata['scale'], plydata['rotation']
        else:
            print('No existing Gaussians from previous layer ...')
       
        identity, _xyz, _features_dc, _features_rest, _scaling, _rotation, _opacity, _identity_mask, _outlier_mask \
            = self.create_from_pcd_and_masks(pcd, masks, spatial_lr_scale)

        # build_dictionary
        if plydata is not None:
            print("Building active mask and surface mask...")
            xyz2, active_mask, surface_mask = self.get_active_mask(plydata, pcd, _identity_mask, _outlier_mask)
            # occluded_mask = self.get_self_occluded_gaussian_mask(plydata)
            # print('======================!!!!!!!!!!!!!!!==============Number of occluded Gaussisans',np.sum(occluded_mask))
            
            # active_mask = np.load(path.replace('point_cloud.ply', 'active_mask.npy'))
            # surface_mask = np.load(path.replace('point_cloud.ply', 'surface_mask.npy'))
            active_mask[:] = False
            static_mask = ~active_mask
            self.n_pcd_points = _xyz.shape[0]
            print("Number of points at initialisation : ", _xyz.shape, xyz2.shape, identity.shape, active_mask.shape)
            _xyz = torch.cat([_xyz, torch.tensor(xyz2[active_mask], dtype=torch.float, device="cuda")], dim=0)
            _features_dc = torch.cat([_features_dc, torch.tensor(features_dc[active_mask], dtype=torch.float, device="cuda").contiguous()], dim=0)
            _features_rest = torch.cat([_features_rest, torch.tensor(features_extra[active_mask], dtype=torch.float, device="cuda").contiguous()], dim=0)
            _scaling = torch.cat([_scaling, torch.tensor(scales[active_mask], dtype=torch.float, device="cuda")], dim=0)
            _rotation = torch.cat([_rotation, torch.tensor(rots[active_mask], dtype=torch.float, device="cuda")], dim=0)
            # opacities[active_mask] = -3
            
            _opacity = torch.cat([_opacity, torch.tensor(opacities[active_mask], dtype=torch.float, device="cuda")], dim=0)
            identity_active = torch.ones((xyz[active_mask].shape[0]), dtype=torch.float, device="cuda")
            identity_active[:] = 2.0
            identity = torch.cat([identity, identity_active], dim=0)

        ## Original 
        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Static Gaussians = Surface Gaussians + Occluded Gaussians (by new pcd points)
        if plydata is not None:
            # print("Number of static points: ", xyz2[static_mask].shape[0])
            self._xyz_static = nn.Parameter(torch.tensor(xyz2[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc_static = nn.Parameter(torch.tensor(features_dc[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._features_rest_static = nn.Parameter(torch.tensor(features_extra[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._opacity_static = nn.Parameter(torch.tensor(opacities[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling_static = nn.Parameter(torch.tensor(scales[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation_static = nn.Parameter(torch.tensor(rots[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))

            self._xyz_surface = nn.Parameter(torch.tensor(xyz2[surface_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc_surface = nn.Parameter(torch.tensor(features_dc[surface_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._features_rest_surface = nn.Parameter(torch.tensor(features_extra[surface_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
            self._opacity_surface = nn.Parameter(torch.tensor(opacities[surface_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling_surface = nn.Parameter(torch.tensor(scales[surface_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation_surface = nn.Parameter(torch.tensor(rots[surface_mask], dtype=torch.float, device="cuda").requires_grad_(True))
            self.identity_surface = torch.zeros((xyz2[surface_mask].shape[0]), device="cuda")
            self.identity = torch.cat([self.identity_surface, identity], dim=0)
        else:

            self._xyz_surface = self._xyz_surface.to(self._xyz.device)
            self._features_dc_surface = self._features_dc_surface.to(self._features_dc.device)
            self._features_rest_surface = self._features_rest_surface.to(self._features_rest.device)
            self._opacity_surface = self._opacity_surface.to(self._opacity.device)
            self._scaling_surface = self._scaling_surface.to(self._scaling.device)
            self._rotation_surface = self._rotation_surface.to(self._rotation.device)

            self._xyz_static = self._xyz_static.to(self._xyz.device)
            self._features_dc_static = self._features_dc_static.to(self._features_dc.device)
            self._features_rest_static = self._features_rest_static.to(self._features_rest.device)
            self._opacity_static = self._opacity_static.to(self._opacity.device)
            self._scaling_static = self._scaling_static.to(self._scaling.device)
            self._rotation_static = self._rotation_static.to(self._rotation.device)
            self.identity_surface = torch.empty(0).to(self._xyz.device)
            self.identity = torch.cat([self.identity_surface, identity], dim=0)


        ## for PSNR
        # surface_mask = surface_mask & (~occluded_mask)
        # if plydata is not None:
        #     static_mask = static_mask & (~surface_mask)
        #     static_mask = static_mask | occluded_mask
        #     self._xyz_static = nn.Parameter(torch.tensor(xyz2[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        #     self._features_dc_static = nn.Parameter(torch.tensor(features_dc[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        #     self._features_rest_static = nn.Parameter(torch.tensor(features_extra[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        #     self._opacity_static = nn.Parameter(torch.tensor(opacities[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        #     self._scaling_static = nn.Parameter(torch.tensor(scales[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        #     self._rotation_static = nn.Parameter(torch.tensor(rots[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))

        #     _xyz_new = torch.cat([_xyz, torch.tensor(xyz2[surface_mask], dtype=torch.float, device="cuda")], dim=0)
        #     _features_dc_new = torch.cat([_features_dc, torch.tensor(features_dc[surface_mask], dtype=torch.float, device="cuda")], dim=0)
        #     _features_rest_new = torch.cat([_features_rest, torch.tensor(features_extra[surface_mask], dtype=torch.float, device="cuda")], dim=0)
        #     _scaling_new = torch.cat([_scaling, torch.tensor(scales[surface_mask], dtype=torch.float, device="cuda")], dim=0)
        #     _rotation_new = torch.cat([_rotation, torch.tensor(rots[surface_mask], dtype=torch.float, device="cuda")], dim=0)
        #     _opacity_new = torch.cat([_opacity, torch.tensor(opacities[surface_mask], dtype=torch.float, device="cuda")], dim=0)
            
        #     self._xyz = nn.Parameter(_xyz_new.requires_grad_(True))
        #     self._features_dc = nn.Parameter(_features_dc_new.requires_grad_(True))
        #     self._features_rest = nn.Parameter(_features_rest_new.requires_grad_(True))
        #     self._scaling = nn.Parameter(_scaling_new.requires_grad_(True))
        #     self._rotation = nn.Parameter(_rotation_new.requires_grad_(True))
        #     self._opacity = nn.Parameter(_opacity_new.requires_grad_(True))
        #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        #     identity_new = torch.ones((xyz2[surface_mask].shape[0]), dtype=torch.float, device="cuda")


        #     self._xyz_surface = self._xyz_surface.to(self._xyz.device)
        #     self._features_dc_surface = self._features_dc_surface.to(self._features_dc.device)
        #     self._features_rest_surface = self._features_rest_surface.to(self._features_rest.device)
        #     self._opacity_surface = self._opacity_surface.to(self._opacity.device)
        #     self._scaling_surface = self._scaling_surface.to(self._scaling.device)
        #     self._rotation_surface = self._rotation_surface.to(self._rotation.device)
        #     self.identity_surface = torch.empty(0).to(self._xyz.device)
        #     self.identity = torch.cat([self.identity_surface, identity, identity_new], dim=0)        
        
        # else:
        #     self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        #     self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        #     self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        #     self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        #     self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        #     self._opacity = nn.Parameter(_opacity.requires_grad_(True)) 
        #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #     self._xyz_surface = self._xyz_surface.to(self._xyz.device)
        #     self._features_dc_surface = self._features_dc_surface.to(self._features_dc.device)
        #     self._features_rest_surface = self._features_rest_surface.to(self._features_rest.device)
        #     self._opacity_surface = self._opacity_surface.to(self._opacity.device)
        #     self._scaling_surface = self._scaling_surface.to(self._scaling.device)
        #     self._rotation_surface = self._rotation_surface.to(self._rotation.device)
            

        #     self._xyz_static = self._xyz_static.to(self._xyz.device)
        #     self._features_dc_static = self._features_dc_static.to(self._features_dc.device)
        #     self._features_rest_static = self._features_rest_static.to(self._features_rest.device)
        #     self._opacity_static = self._opacity_static.to(self._opacity.device)
        #     self._scaling_static = self._scaling_static.to(self._scaling.device)
        #     self._rotation_static = self._rotation_static.to(self._rotation.device)
        #     self.identity_surface = torch.empty(0).to(self._xyz.device)
        #     self.identity = torch.cat([self.identity_surface, identity], dim=0)

        print('Number of static and surface points at initialization:', len(self._xyz_static), len(self._xyz_surface))
        self.active_sh_degree = self.max_sh_degree

    def load_gaussians(self, plydata, spatial_lr_scale):
        # plydata = {'xyz':xyz, 'normals':normals, 'f_dc':f_dc, 'f_rest':f_rest, 'opacities':opacities, 'scale':scale, 'rotation':rotation}
        if plydata is not None:
            xyz, opacities, features_dc, features_extra, scales, rots = plydata['xyz'], plydata['opacities'], plydata['f_dc'], plydata['f_rest'], plydata['scale'], plydata['rotation']
        else:
            print('No existing Gaussians from previous layer ...')
       
        # build_dictionary
        print("Building occlusion mask...")
        occluded_mask = self.get_self_occluded_gaussian_mask(plydata)
        # print('======================!!!!!!!!!!!!!!!==============Number of occluded Gaussisans',np.sum(occluded_mask))
        
        # active_mask = np.load(path.replace('point_cloud.ply', 'active_mask.npy'))
        # surface_mask = np.load(path.replace('point_cloud.ply', 'surface_mask.npy'))
        active_mask = ~occluded_mask
        static_mask = occluded_mask

        # print("Number of points at initialisation : ", _xyz.shape, identity.shape, active_mask.shape)
        _xyz = torch.tensor(xyz[active_mask], dtype=torch.float, device="cuda")
        _features_dc = torch.tensor(features_dc[active_mask], dtype=torch.float, device="cuda").contiguous()
        _features_rest = torch.tensor(features_extra[active_mask], dtype=torch.float, device="cuda").contiguous()
        _scaling = torch.tensor(scales[active_mask], dtype=torch.float, device="cuda")
        _rotation = torch.tensor(rots[active_mask], dtype=torch.float, device="cuda")
        # opacities[active_mask] = -3
        
        _opacity = torch.tensor(opacities[active_mask], dtype=torch.float, device="cuda")
        identity_active = torch.ones((xyz[active_mask].shape[0]), dtype=torch.float, device="cuda")
        identity_active[:] = 2.0
        identity = identity_active

        ## Original 
        self._xyz = nn.Parameter(_xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(_features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(_features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(_scaling.requires_grad_(True))
        self._rotation = nn.Parameter(_rotation.requires_grad_(True))
        self._opacity = nn.Parameter(_opacity.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Static Gaussians = Surface Gaussians + Occluded Gaussians (by new pcd points)
        
        self._xyz_static = nn.Parameter(torch.tensor(xyz[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc_static = nn.Parameter(torch.tensor(features_dc[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._features_rest_static = nn.Parameter(torch.tensor(features_extra[static_mask], dtype=torch.float, device="cuda").contiguous().requires_grad_(True))
        self._opacity_static = nn.Parameter(torch.tensor(opacities[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling_static = nn.Parameter(torch.tensor(scales[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_static = nn.Parameter(torch.tensor(rots[static_mask], dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz_surface = self._xyz_surface.to(self._xyz.device)
        self._features_dc_surface = self._features_dc_surface.to(self._features_dc.device)
        self._features_rest_surface = self._features_rest_surface.to(self._features_rest.device)
        self._opacity_surface = self._opacity_surface.to(self._opacity.device)
        self._scaling_surface = self._scaling_surface.to(self._scaling.device)
        self._rotation_surface = self._rotation_surface.to(self._rotation.device)
        self.identity_surface = torch.empty(0).to(self._xyz.device)
        self.identity = identity    

        print('Number of static and surface points at finetune:', len(self._xyz_static), len(self._xyz))
        self.active_sh_degree = self.max_sh_degree

    def training_setup(self, training_args):
        self.update_identity_mask()

        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # def save_ply(self, filepath):
    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(filepath)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz_total.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat((self._features_dc_static, self._features_dc),dim=0)
        f_dc = f_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = torch.cat((self._features_rest_static, self._features_rest),dim=0)
        f_rest = f_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest[:]  = 0
        opacities = torch.cat([self._opacity_static, self._opacity], dim=0)
        opacities = opacities.detach().cpu().numpy()
        _scaling = torch.cat([self._scaling_static, self._scaling], dim=0)
        scale = _scaling.detach().cpu().numpy()
        _rotation = torch.cat([self._rotation_static, self._rotation], dim=0)
        rotation = _rotation.detach().cpu().numpy()


        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        # identity = self.identity.detach().cpu()
        # torch.save(identity, path.replace('.ply', '_identity.pt'))

        # print(path, self._xyz_static.shape, xyz.shape)
        # if 'layer0' not in path:
        #     path2 = path.replace('.ply', '_static.ply')
        #     xyz = self._xyz_static.detach().cpu().numpy()
        #     normals = np.zeros_like(xyz)
        #     f_dc = self._features_dc_static
        #     f_dc = f_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #     f_rest = self._features_rest_static
        #     f_rest = f_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #     f_rest[:]  = 0
        #     opacities = self._opacity_static
        #     opacities = opacities.detach().cpu().numpy()
        #     _scaling = self._scaling_static
        #     scale = _scaling.detach().cpu().numpy()
        #     _rotation = self._rotation_static
        #     rotation = _rotation.detach().cpu().numpy()


        #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
        #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        #     elements[:] = list(map(tuple, attributes))
        #     el = PlyElement.describe(elements, 'vertex')
        #     PlyData([el]).write(path2)


    def wrap_gaussian(self):
        # unlike save_ply, we don't transpose the sh degrees to the last dim for feature dc and rest
        xyz = self.get_xyz_total.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = torch.cat((self._features_dc_static, self._features_dc),dim=0)
        print("DEBUG wrap gaussian shape,static,active",self._features_dc_static.shape, self._features_dc.shape)
        f_dc = f_dc.detach().contiguous().cpu().numpy()
        f_rest = torch.cat((self._features_rest_static, self._features_rest),dim=0)
        f_rest = f_rest.detach().contiguous().cpu().numpy()
        opacities = torch.cat([self._opacity_static, self._opacity], dim=0)
        opacities = opacities.detach().cpu().numpy()
        _scaling = torch.cat([self._scaling_static, self._scaling], dim=0)
        scale = _scaling.detach().cpu().numpy()
        _rotation = torch.cat([self._rotation_static, self._rotation], dim=0)
        rotation = _rotation.detach().cpu().numpy()

        # xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()
        print('Gaussians Wrapped with {} points.'.format(len(xyz)))
        return {'xyz':xyz, 'normals':normals, 'f_dc':f_dc, 'f_rest':f_rest, 'opacities':opacities, 'scale':scale, 'rotation':rotation}
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(filepath)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz_surface = self._xyz_surface.to(self._xyz.device)
        self._features_dc_surface = self._features_dc_surface.to(self._features_dc.device)
        self._features_rest_surface = self._features_rest_surface.to(self._features_rest.device)
        self._opacity_surface = self._opacity_surface.to(self._opacity.device)
        self._scaling_surface = self._scaling_surface.to(self._scaling.device)
        self._rotation_surface = self._rotation_surface.to(self._rotation.device)

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.update_identity_mask()
        self.identity = torch.cat([self.identity_surface, self.identity[self.identity_mask][valid_points_mask]], dim=0)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        self.identity = torch.cat((self.identity, self.identity[self.identity_mask][selected_pts_mask].repeat(N)), dim=0)
        self.update_identity_mask()

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.identity = torch.cat((self.identity, self.identity[self.identity_mask][selected_pts_mask]), dim=0)
        self.update_identity_mask()

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        self.update_identity_mask()
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        self.update_identity_mask()

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, identity_mask):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[identity_mask][update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


    def get_active_mask(self, plydata, pcd, identity_mask0, outlier_mask):

        xyz, scales = plydata['xyz'], plydata['scale']     
        scales_max = np.exp(scales)
        scales_max = scales_max.max(axis=1)
        # print(scales_max.shape)
        _xyz = np.asarray(pcd.points)[identity_mask0][outlier_mask]
        
        scale_func = lambda x: np.log(x+1) * 200
        # build_dictionary
        rays_d = []
        # z_table.fill(zmax)
        for i in tqdm(range(len(_xyz))):
            dist = np.sqrt(_xyz[i,0]**2 + _xyz[i,1]**2 + _xyz[i,2]**2)
            dx = _xyz[i,0] / dist
            dy = _xyz[i,1] / dist
            dz = _xyz[i,2] / dist
            rays_d.append([dx,dy,dz,dist]) 

        rays_d = np.array(rays_d)

        # print(rays_d[:,0].max(), rays_d[:,0].min())
        # print(rays_d[:,1].max(), rays_d[:,1].min())
        # print(rays_d[:,2].max(), rays_d[:,2].min())

        xmin, xmax = rays_d[:,0].min(), rays_d[:,0].max()
        ymin, ymax = rays_d[:,1].min(), rays_d[:,1].max()
        zmin, zmax = rays_d[:,2].min(), rays_d[:,2].max()

        x_range = scale_func(xmax - xmin) 
        y_range = scale_func(ymax - ymin) 
        z_range = scale_func(zmax - zmin)
        dist_voxel = np.zeros(((np.ceil(x_range).astype('int'))+1, np.ceil(y_range).astype('int')+1, np.ceil(z_range).astype('int')+1))
        dist_occupancy = np.zeros(((np.ceil(x_range).astype('int'))+1, np.ceil(y_range).astype('int')+1, np.ceil(z_range).astype('int')+1), dtype=bool)
        for i in tqdm(range(len(rays_d))):
            x = np.ceil(scale_func(rays_d[i,0] - xmin)).astype('int')
            y = np.ceil(scale_func(rays_d[i,1] - ymin)).astype('int')
            z = np.ceil(scale_func(rays_d[i,2] - zmin)).astype('int')
            # print(scale_func(rays_d[i,0] - xmin), x)
            dist_occupancy[x,y,z] = True
            if dist_voxel[x,y,z] < rays_d[i,3]:
                dist_voxel[x,y,z] = rays_d[i,3]

        # print(dist_voxel.shape)
        
        pcd_mask = np.zeros((xyz.shape[0]), dtype=bool)
        surface_mask = np.ones((xyz.shape[0]), dtype=bool)
        for idx in tqdm(range(len(xyz))):
            dist = np.sqrt(xyz[idx,0]**2 + xyz[idx,1]**2 + xyz[idx,2]**2)
            
            dx = xyz[idx,0] / dist
            dy = xyz[idx,1] / dist
            dz = xyz[idx,2] / dist

            dist = dist - scales_max[idx]
            x = np.ceil(scale_func(dx - xmin)).astype('int')
            y = np.ceil(scale_func(dy - ymin)).astype('int')
            z = np.ceil(scale_func(dz - zmin)).astype('int')

            if x >= 0 and x < np.ceil(x_range).astype('int')+1 and y >= 0 and y < np.ceil(y_range).astype('int')+1 and z >= 0 and z < np.ceil(z_range).astype('int')+1:
                if dist <= dist_voxel[x,y,z]:
                    pcd_mask[idx] = True
                    # xyz[idx,2] = dist_voxel[x,y,z]
                    surface_mask[idx] = False # type3 - active gaussian
                elif dist_occupancy[x,y,z]:
                    pcd_mask[idx] = False
                    surface_mask[idx] = False # type1 - occluded gaussians
                else:
                    pcd_mask[idx] = False # type 2 - surface gaussians
            else:
                pcd_mask[idx] = False # type 2 - out-of-domain / surface gaussians

        

        return xyz, pcd_mask, surface_mask

    def get_self_occluded_gaussian_mask(self, plydata):

        xyz, scales = plydata['xyz'], plydata['scale']     
        scales_max = np.exp(scales)
        scales_max = scales_max.max(axis=1)
        # print(scales_max.shape)
        
        scale_func = lambda x: np.log(x+1) * 200
        rays_d = []
        for i in tqdm(range(len(xyz))):
            dist = np.sqrt(xyz[i,0]**2 + xyz[i,1]**2 + xyz[i,2]**2)
            dx = xyz[i,0] / dist
            dy = xyz[i,1] / dist
            dz = xyz[i,2] / dist
            rays_d.append([dx,dy,dz,dist]) 

        rays_d = np.array(rays_d)

        xmin, xmax = rays_d[:,0].min(), rays_d[:,0].max()
        ymin, ymax = rays_d[:,1].min(), rays_d[:,1].max()
        zmin, zmax = rays_d[:,2].min(), rays_d[:,2].max()

        x_range = scale_func(xmax - xmin) 
        y_range = scale_func(ymax - ymin) 
        z_range = scale_func(zmax - zmin)
        dist_voxel = np.zeros(((np.ceil(x_range).astype('int'))+1, np.ceil(y_range).astype('int')+1, np.ceil(z_range).astype('int')+1))
        dist_voxel[:,:,:] = 1000
        # dist_occupancy = np.zeros(((np.ceil(x_range).astype('int'))+1, np.ceil(y_range).astype('int')+1, np.ceil(z_range).astype('int')+1), dtype=bool)
        for i in tqdm(range(len(rays_d))):
            x = np.ceil(scale_func(rays_d[i,0] - xmin)).astype('int')
            y = np.ceil(scale_func(rays_d[i,1] - ymin)).astype('int')
            z = np.ceil(scale_func(rays_d[i,2] - zmin)).astype('int')
            # print(scale_func(rays_d[i,0] - xmin), x)
            # dist_occupancy[x,y,z] = True
            if dist_voxel[x,y,z] > rays_d[i,3]:
                dist_voxel[x,y,z] = rays_d[i,3]
                

        # print(dist_voxel.shape)
        
        # pcd_mask = np.zeros((xyz.shape[0]), dtype=bool)
        occluded_mask = np.zeros((xyz.shape[0]), dtype=bool)
        for idx in tqdm(range(len(xyz))):
            dist = np.sqrt(xyz[idx,0]**2 + xyz[idx,1]**2 + xyz[idx,2]**2)
            
            dx = xyz[idx,0] / dist
            dy = xyz[idx,1] / dist
            dz = xyz[idx,2] / dist

            # dist = dist - scales_max[idx]
            x = np.ceil(scale_func(dx - xmin)).astype('int')
            y = np.ceil(scale_func(dy - ymin)).astype('int')
            z = np.ceil(scale_func(dz - zmin)).astype('int')
            # print(x,y,z, x_range, y_range, z_range)
            if x >= 0 and x < np.ceil(x_range).astype('int')+1 and y >= 0 and y < np.ceil(y_range).astype('int')+1 and z >= 0 and z < np.ceil(z_range).astype('int')+1:
                # print(dist, dist_voxel[x,y,z])
                # print(dist, dist_voxel[x,y,z])
                if dist > dist_voxel[x,y,z]+2:
                    
                    occluded_mask[idx] = True
               
        # np.save(gaussian_path.replace('point_cloud.ply', 'active_mask.npy'), pcd_mask)
        # np.save(gaussian_path.replace('point_cloud.ply', 'surface_mask.npy'), surface_mask)

        return occluded_mask

    def get_outlier_filter(self, points, edge_mask,thresh=3):
        scale = 1
        xmin, xmax = points[:,0].min(), points[:,0].max()
        ymin, ymax = points[:,1].min(), points[:,1].max()
        zmin, zmax = points[:,2].min(), points[:,2].max()

        x_range = (xmax - xmin) * scale
        y_range = (ymax - ymin) * scale
        z_range = (zmax - zmin) * scale
        z_table = np.zeros((int(x_range)+1, int(y_range)+1, int(z_range)+1), dtype=np.uint8)
        for i in tqdm(range(len(points))):
            x = int((points[i,0] - xmin) * scale)
            y = int((points[i,1] - ymin) * scale)
            z = int((points[i,2] - zmin) * scale)
            z_table[x,y,z] += 1

        point_mask = np.zeros((len(points)), dtype=bool)
        for i in tqdm(range(len(points))):
            x = int((points[i,0] - xmin) * scale)
            y = int((points[i,1] - ymin) * scale)
            z = int((points[i,2] - zmin) * scale)
            if z_table[x,y,z] < thresh:
                point_mask[i] = True
        # point_mask = point_mask & edge_mask
        point_mask = ~point_mask
        return point_mask


    def get_outlier_filter2(self, points, edge_mask, thresh=3):
        # Example: Create an array of points where each point is in 3D space.
        edge_points = points[edge_mask]
        print('Number of edge points',len(edge_points))
        # Build a k-d tree from these points
        tree = cKDTree(edge_points)

        # Specify the query point and the radius
        radius = 0.01  # Change this value to your specific radius

        point_mask = np.zeros((len(edge_points)), dtype=bool)
        for idx in tqdm(range(len(edge_points))):
            # Query the tree for points within the radius of the query_point
            indices = tree.query_ball_point(edge_points[idx], radius)

            # The result, 'indices', is a list of all points within the radius
            number_of_neighbors = len(indices)
            if number_of_neighbors < thresh:
                point_mask[idx] = True
                
        print('Number of outliers',point_mask.sum())
        new_mask = edge_mask.copy()
        edge_indices = np.where(edge_mask)[0]
        new_mask[edge_indices] = point_mask
        print(edge_mask.sum(), new_mask.sum())
        return ~new_mask
    
def save_pcd(filename, points, colors):
    """
    Save a point cloud to a PLY file.

    Parameters:
    - filename: str, the name of the output PLY file.
    - points: numpy.ndarray, an Nx3 array containing the XYZ coordinates of the points.
    - colors: numpy.ndarray, an Nx3 array containing the RGB colors of the points.

    Returns:
    - None
    """
    # Ensure points and colors are numpy arrays
    points = np.asarray(points)
    colors = np.asarray(colors*255, dtype=np.uint8)

    # Check that points and colors have the same number of rows
    if points.shape[0] != colors.shape[0]:
        raise ValueError("The number of points must match the number of colors")

    # Create a structured array with points and colors
    vertex_data = np.array([(points[i, 0], points[i, 1], points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]) 
                            for i in range(points.shape[0])],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Create a PlyElement
    vertex_element = PlyElement.describe(vertex_data, 'vertex')

    # Write the PLY file
    PlyData([vertex_element], text=True).write(filename)
    print(f"Point cloud saved to {filename}")