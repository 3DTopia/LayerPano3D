import os

from PIL import Image
from argparse import ArgumentParser
import numpy as np
import math

parser = ArgumentParser(description="Step 0 Parameters")

parser.add_argument("--name", default="", type=str)
parser.add_argument("--n_layer", default=1, type=int)
parser.add_argument("--focal_scale", default=1.0, type=float)
args = parser.parse_args()

base_path = r'E:\Code\depth_alignment\{}'.format(args.name)
# os.makedirs(base_path, exist_ok=True)
n_layers = args.n_layer


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


base_img = Image.open(os.path.join(base_path, 'layer{}_inpaint.png'.format(n_layers)))
height, width = base_img.height, base_img.width
focal = np.sqrt(height**2 + width**2) / args.focal_scale
ours_FovY = focal2fov(focal, height)
ours_FovX = focal2fov(focal, width)
ours_intrinsics = get_intrinsics2(height, width, ours_FovX, ours_FovY)
np.save(os.path.join(base_path, "intrinsics.npy"),ours_intrinsics)

# os.system('cp {}/sample.png {}/layer{}_inpaint.png'.format(args.name, base_path, n_layers))
# os.system('cd depth_inpainting/run')
for l in range(n_layers, 0, -1):
    input_rgb_path=os.path.join(base_path, 'layer{}_inpaint.png'.format(l-1))
    input_mask_path=os.path.join(base_path, "layer{}_mask_smooth.png".format(l))
    input_depth_path=os.path.join(base_path, "layer{}_inpaint_depth.npy".format(l))
    intri=os.path.join(base_path, "intrinsics.npy")
    output_dir= base_path
    image_path = os.path.join(base_path, 'layer{}_inpaint.png'.format(l))
    os.system(f'python run_inference_inpainting.py --input_rgb_path {input_rgb_path} --input_mask {input_mask_path} --input_depth_path {input_depth_path} --output_dir {output_dir} --intri {intri} --use_mask  --blend')