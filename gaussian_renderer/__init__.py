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

import torch
import math

# from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings4d
# from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer4d

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


# from diff_gaussian_rasterization_ys import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.layer_gaussian_model import LayerGaussian
from utils.sh import eval_sh

def render(viewpoint_camera, 
           pc: LayerGaussian, 
           opt, 
           bg_color: torch.Tensor, 
           scaling_modifier=1.0, 
           override_color=None, 
           render_only=False):
    

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz_render, dtype=pc.get_xyz_render.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz_render
    means2D = screenspace_points
    opacity = pc.get_opacity_render

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance_render(scaling_modifier)
    else:
        scales = pc.get_scaling_render
        rotations = pc.get_rotation_render




    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            n_points = len(pc.get_xyz_render)
            shs_view = pc.get_features_render.transpose(1, 2).view(n_points, 3, -1)[:,:,:(pc.max_sh_degree+1)**2]
            dir_pp = (pc.get_xyz_render - viewpoint_camera.camera_center.repeat(pc.get_features_render.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features_render[:,:,:(pc.max_sh_degree+1)**2]
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, mask = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if render_only:
        return {"render": rendered_image, "depth": depth, "mask": mask}
    else:
        return {"render": rendered_image,
                "depth": depth,
                "mask": mask,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}


def render4d(viewpoint_camera, 
             pc: GaussianModel, 
             opt, 
             bg_color: torch.Tensor, 
             scaling_modifier=1.0, 
             override_color=None, 
             render_only=False,
             stage="fine"):

    from diff_gaussian_rasterization import GaussianRasterizationSettings  as GaussianRasterizationSettings4d
    from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer4d
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, 
                                          dtype=pc.get_xyz.dtype, 
                                          requires_grad=True, 
                                          device="cuda") + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings4d(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=opt.debug
    )
    rasterizer = GaussianRasterizer4d(raster_settings=raster_settings)
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    time = (time.float()  - 4) / 10 # hack
    # time = time.float() / 10

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if opt.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # scales = pc.get_scaling
        # rotations = pc.get_rotation
        scales = pc._scaling
        rotations = pc._rotation

    deformation_point = pc._deformation_table  ##add
    deformation_point = deformation_point.bool()

    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(
                                                                         means3D[deformation_point], 
                                                                         scales[deformation_point], 
                                                                         rotations[deformation_point], 
                                                                         opacity[deformation_point],
                                                                         time[deformation_point])


    # print("deformation_point", deformation_point.shape, type(deformation_point))
    # print(deformation_point.dtype)
    with torch.no_grad():
        pc._deformation_accum[deformation_point] += torch.abs(means3D_deform-means3D[deformation_point])


    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if opt.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
        
    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    
    rendered_image = rendered_image.clamp(0, 1)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if render_only:
        return {"render": rendered_image, "depth": depth}
    else:
        return {"render": rendered_image,
                "depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}