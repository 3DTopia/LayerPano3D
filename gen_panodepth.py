import os
import sys 
import cv2
import numpy as np
import argparse
import torch.nn.functional as F

from PIL import Image
from utils.utils_basic import make_bagel_mask, icosahedron_sample_camera
from utils.trajectory import gcd_pose_gs

import utils.pano_utils.Equirec2Perspec as E2P
import utils.pano_utils.multi_Perspec2Equirec as m_P2E
python_src_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Adding '{python_src_dir}' to sys.path") 
sys.path.append(python_src_dir + "/submodules/360monodepth/code/python/src/") 

from utility import depthmap_utils, pointcloud_utils
from utils.depth_alignment import depth_alignment, Pano_depth_estimation




class Gen_panodepth:
    def __init__(self, depth_model='DepthAnythingv2', save_dir='outputs_panodepth', input_path='test.png' , pathtxt=None , device='cuda'):
        self.save_dir = save_dir
        self.device = device
        self.depth_model = depth_model
        self.pathtxt = pathtxt
        self.input_path = input_path
        os.makedirs(self.save_dir, exist_ok=True)

    def read_paths(self, file_path):
        rgbpath_list = []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    path = line.strip() 
                    if path:  
                        rgbpath_list.append(path)
        except FileNotFoundError:
            print(f"file {file_path} not found")
        except Exception as e:
            print(f"error in reading: {e}")
        
        return rgbpath_list

    def readImg(self, path):
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        return img

    def generate_panodepth(self, pano_rgb):
        ####### pano_rgb        [pano_h,pano_w,3] [0-255]        
        pano_h, pano_w = pano_rgb.shape[0], pano_rgb.shape[1]
        panodepth_estimator = Pano_depth_estimation(pano_h, pano_w, self.save_dir, self.device, depth_model=self.depth_model)
        pano_depth = panodepth_estimator.get_panodepth(pano_rgb)  #[0-1] 

        return pano_depth  


    def run(self):
        pano_rgb_list = []
        pano_depth_list = []
        max_val = 65535
        if self.pathtxt is not None:
            pano_rgb_path = self.read_paths(self.pathtxt)
        else:
            pano_rgb_path = [self.input_path]
        N = len(pano_rgb_path)
        for i in range(N):
            pano_rgb = self.readImg(pano_rgb_path[i])
            pano_depth = self.generate_panodepth(pano_rgb)
            
            if N==1:
                save_dir_now = f"{self.save_dir}/layering"
            else:
                save_dir_now = f'{self.save_dir}/layering/{i}'
            os.makedirs(save_dir_now, exist_ok=True)
            rgb_path = f'{save_dir_now}/rgb.png'
            depth_path = f'{save_dir_now}/depth.png'
            np.save(f'{save_dir_now}/depth.npy', pano_depth)
            depthmap_utils.depth_visual_save(pano_depth, f'{save_dir_now}/depth_rgb.png')
            pointcloud_utils.depthmap2pointcloud_erp(pano_depth , pano_rgb, f"{save_dir_now}/pcd_rgb.ply")



            rgb_pil = Image.fromarray(pano_rgb)
            rgb_pil.save(rgb_path)

            
            pano_depth = (pano_depth -  pano_depth.min()) / (pano_depth.max() - pano_depth.min())
            
            depth = (pano_depth * max_val).astype(np.uint16)
            depth_pil = Image.fromarray(depth)
            depth_pil.save(depth_path)



parser = argparse.ArgumentParser(description='Arguments for PanoScene4D')
parser.add_argument('--depth_model', type=str, default="DepthAnythingv2")
parser.add_argument('--save_dir', type=str, default="outputs")
parser.add_argument('--input_path', type=str, default='outputs/rgb.png') 

args = parser.parse_args()
depth_model = args.depth_model
save_dir = args.save_dir
input_path = args.input_path

gen_panodepth = Gen_panodepth(depth_model = depth_model,save_dir=save_dir, input_path=input_path)
gen_panodepth.run() #DepthAnythingv2, zoedepth