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
from scene import Scene
from scene.cameras import Camera
import os
import numpy as np
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])
def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    test_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "test_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    depth_dis_path=os.path.join(model_path, name, "ours_{}".format(iteration), "depth_dis")
    intri_path=os.path.join(model_path, name, "ours_{}".format(iteration), "intri")
    c2w_path=os.path.join(model_path, name, "ours_{}".format(iteration), "c2w")
    makedirs(render_path, exist_ok=True)
    makedirs(test_render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_dis_path, exist_ok=True)
    makedirs(intri_path, exist_ok=True)
    makedirs(c2w_path, exist_ok=True)
    print(gaussians._xyz.shape)
    n=len(views)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        renders = render(view, gaussians, pipeline, background)
        c2w=torch.eye(4)
        c2w[:3, :3]=torch.from_numpy(view.R)
        c2w[:3, 3] = -torch.from_numpy(view.R) @ torch.from_numpy(view.T)
        depth=renders["depth_3dgs"].squeeze(0).detach().cpu().numpy()
        print(depth.min())
        depth_dis=np.reciprocal(depth)
        intrinsics = get_intrinsics2(view.image_height, view.image_width,view.FoVx,view.FoVy)
        np.save(os.path.join(c2w_path, view.image_name+".npy"),c2w)
        np.save(os.path.join(intri_path, view.image_name+".npy"),intrinsics)
        np.save(os.path.join(depth_path, view.image_name+".npy"),depth)
        np.save(os.path.join(depth_dis_path, view.image_name+".npy"),depth_dis)
        rendering=renders["render"]
        visibility_filter, radii =renders["visibility_filter"],renders["radii"]
        gt = view.original_image[0:3, :, :]
        img=depth
        img = (img - img.min()) / (img.max() - img.min())
        img = (img * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(depth_path, view.image_name + ".png"), img)
        torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)