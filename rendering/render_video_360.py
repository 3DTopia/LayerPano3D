import os
import cv2
import glob
import json

import warnings

from random import randint
from argparse import ArgumentParser
from loguru import logger
warnings.filterwarnings(action='ignore')

import imageio
import shutil
import random

import numpy as np
import math
from PIL import Image
from tqdm import tqdm

import torch


from arguments import GSParams
from gaussian_renderer import render
from scene import  LayerGaussian

from utils.depth_utils import colorize
from scene.cameras import  MiniCam_GS

from tqdm import tqdm

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def camera_translation(current, destination, n_poses=360):
    """
    Generate a flythrough path for rendering
    Inputs:
        depth: float, the depth that of the flythrough
        n_poses: int, number of poses to create along the path
    Params:
        R: (3x3) the camera pose of the starting position
        T: (3) the coordinate of the starting position
    Outputs:
        poses_flythrough: [(T, R)] the cam to world transformation matrix of a spiral path
    """

    flythrough_cams = []
    R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    current_z, current_x = current
    destination_z, destination_x = destination
    if current_z <  destination_z:
        steps_z = np.linspace(current_z, destination_z, n_poses + 1)[:-1]
    else:
        steps_z = np.linspace(destination_z, current_z, n_poses + 1)[:-1][::-1]

    if current_x <  destination_x:
        steps_x = np.linspace(current_x, destination_x, n_poses + 1)[:-1]
    else:
        steps_x = np.linspace(destination_x, current_x, n_poses + 1)[:-1][::-1]

    # print(current_z, current_x, destination_z, destination_x)
    for i in range(len(steps_z)):
        new_T = [steps_x[i], 0, -steps_z[i]]
        flythrough_cams += [(new_T, R)]
    # for step in np.linspace(0, depth, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
    #     new_T = [0, 0, -step]
    #     flythrough_cams += [(new_T, R)]
    return flythrough_cams

def camera_rotation(radius=10, angle=np.pi/9, n_poses=180):
    """
    Generate a 360 degree spherical path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_cams: (n_poses, 3, 4) the cam to world transformation matrix of a circular path
    """

    def circular_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [0, 0, -radius],
        ], dtype=float).T

        rotation_phi = lambda phi: np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ], dtype=float)

        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th)],
            [0, 1, 0],
            [np.sin(th), 0, np.cos(th)],
        ], dtype=float)
        trans_mat = trans_t(radius)
        rot_mat = rotation_phi(phi / 180. * np.pi) 
        rot_mat = rotation_theta(theta) @ rot_mat
        return (trans_mat, rot_mat)

    circular_cams = []
    if angle > 0:
        for th in np.linspace(0, angle, n_poses + 1)[:-1]:
            # print(th)
            circular_cams += [circular_pose(th, 0, radius)]
    elif angle < 0:
        for th in np.linspace(0, -angle, n_poses + 1):
            # print(-th)
            circular_cams += [circular_pose(-th, 0, radius)]
    else:
        for th in range(n_poses):
            circular_cams += [circular_pose(0, 0, radius)]
    
    return circular_cams 

def get_zigzag_trajectory(points, rotations, n_poses=30):
    currents = points[:-1]
    targets = points[1:]
    trajectory = []
    current_R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    for i in range(len(currents)):
        tran_poses = camera_translation(currents[i], targets[i], n_poses=n_poses[i])
        rot_poses = camera_rotation(radius=10, angle=rotations[i], n_poses=n_poses[i])
        print(rotations)
        poses = [(tran_poses[i][0], current_R @ rot_poses[i][1]) for i in range(len(tran_poses))]
        # print(rot_poses)
        trajectory += poses
        current_R = tran_poses[-1][1]
    
    return trajectory

from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description="Step 0 Parameters")
parser.add_argument("--save_dir", default="", type=str)
parser.add_argument("--elevation", default="-45", type=str)
args = parser.parse_args()

save_dir = args.save_dir
layer_names = [i for i in os.listdir(save_dir) if 'gsplat_layer' in i]
print("layer_names:",layer_names)
layer_names.sort()
layer_name = layer_names[-1].split('_')[-1].split('.')[0]
print('Render from {}'.format(layer_name))  
elevation_deg = int(args.elevation)

videopath = f"{args.save_dir}/renders_{elevation_deg}deg.mp4"
depthpath = f"{args.save_dir}/depths_{elevation_deg}deg.mp4"


rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th)],
            [0, 1, 0],
            [np.sin(th), 0, np.cos(th)],
        ], dtype=float)

rotation_phi = lambda phi: np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)],
        ], dtype=float)

phis = np.linspace(0,360,160)
Rs = []
Ts = []

# points = [[0,0], [-30,0], [-50,-20], [-80,0], [0,0], [0,0]] # z,x
# angles = [0, -np.pi/4, np.pi/4, 0, np.pi/2]
# n_poses = [20,30,30,30,30]

pose_dict = {}
starting_view_mat = np.eye(3)
elevation = rotation_phi(elevation_deg / 180. * np.pi)
for idx, phi in enumerate(phis):
    # phi = 90
    rot_pose = rotation_theta(phi / 180. * np.pi) 
    # print(points)

    # zigzag_poses = get_zigzag_trajectory(points, angles, n_poses=n_poses)
    Rs += [starting_view_mat @ rot_pose @ elevation]   
    Ts += [[0,0,0]]
    # points = [[-item[1],item[0]] for item in points]
    
    # starting_view_mat = Rs[-1]

pose_dict['R'] = [rot.tolist() for rot in Rs]
pose_dict['T'] = Ts# [tvec.tolist() for tvec in Ts]

# f = open(os.path.join(save_dir, f'eval_pose_regular.json'), 'w')
# json.dump(pose_dict, sort_keys=True, indent=4, separators=(',', ':'), fp=f)
# f.close()

# poses_dict = load_json(os.path.join(save_dir, f'eval_pose_regular.json'))
Rs = pose_dict['R']
Ts = pose_dict['T']
poses = [(Rs[i], Ts[i]) for i in range(len(Rs))]

views = []

width = height = 2048
fovx = math.radians(90)
fovy = height * fovx / width

# ---------------------------- Settings ---------------------------------#
opt = GSParams()
bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')


for i in range(len(poses)):
    rot, tvec = poses[i]
    rot = np.array(rot)
    tvec = np.array(tvec)
    pose = np.eye(4)
    pose[:3,:3] = rot
    pose[:3,3] = tvec
    # pose = poses[i]
    cur_cam = MiniCam_GS(pose, width, height, fovx, fovy)
    views.append(cur_cam)
    
framelist = []
depthlist = []
dmin, dmax = 1e8, -1e8

with torch.no_grad():
    gaussians = LayerGaussian(opt.sh_degree)
    gaussians.load_ply(os.path.join(save_dir, 'gsplat_{}.ply'.format(layer_name)))
    
    iterable_render = views

    for view in tqdm(iterable_render):
        results = render(view, gaussians, opt, background)
        frame, depth = results['render'], results['depth']
        framelist.append(
            np.round(frame.permute(1,2,0).detach().cpu().numpy().clip(0,1)*255.).astype(np.uint8))
        depth = -(depth * (depth > 0)).detach().cpu().numpy()
        dmin_local = depth.min().item()
        dmax_local = depth.max().item()
        if dmin_local < dmin:
            dmin = dmin_local
        if dmax_local > dmax:
            dmax = dmax_local
        depthlist.append(depth)


sampled_framelist = framelist[:]
# sampled_depthlist = depthlist[::2]
# sampled_depthlist = [colorize(depth) for depth in sampled_depthlist]

# print('Saving sampled frames...')
# for i in range(len(sampled_framelist)):
#     frame_pil = Image.fromarray(sampled_framelist[i])
#     frame_pil.save(os.path.join(framedir,'%4d.png' % (i)))
#     # depth_pil = Image.fromarray(sampled_depthlist[i])
#     # depth_pil.save(os.path.join(depthdir,'%4d.png' % (i)))
# print('Done...')

depthlist = [colorize(depth) for depth in depthlist]

print(len(framelist), len(depthlist))
print('Start Writing Videos...')
imageio.mimwrite(videopath, framelist, fps=20, quality=8)
imageio.mimwrite(depthpath, depthlist, fps=20, quality=8)

print('End Writing Videos...')

