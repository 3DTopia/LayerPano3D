import torch
import numpy as np
import math

from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description="Step 0 Parameters")
parser.add_argument("--focal_scale", default=1.0, type=float)
parser.add_argument("--width", default=1600, type=int)
parser.add_argument("--height", default=1064, type=int)
args = parser.parse_args()

def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

c2w=torch.eye(4)
np.save("c2w.npy",c2w)
height = args.height
width = args.width
focal = np.sqrt(height**2 + width**2) / args.focal_scale
ours_FovY = focal2fov(focal, height)
ours_FovX = focal2fov(focal, width)
ours_intrinsics = get_intrinsics2(height, width, ours_FovX, ours_FovY)
np.save("intrinsics.npy",ours_intrinsics)