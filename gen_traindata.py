import os
import sys 
import cv2
import torch
import numpy as np
import argparse
import torch.nn.functional as F
import time
from PIL import Image
from utils.utils_basic import make_bagel_mask, icosahedron_sample_camera
from utils.trajectory import gcd_pose_gs
from tqdm.auto import tqdm
import matplotlib

import utils.pano_utils.Equirec2Perspec as E2P
import utils.pano_utils.multi_Perspec2Equirec as m_P2E
python_src_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Adding '{python_src_dir}' to sys.path") 
sys.path.append(python_src_dir + "/submodules/360monodepth/code/python/src/") 

from utility import depthmap_utils, pointcloud_utils
from utils.depth_alignment import depth_alignment, Pano_depth_estimation


from src.Infusion.depth_inpainting.inference.depth_inpainting_pipeline_half import DepthEstimationInpaintPipeline
from src.Infusion.depth_inpainting.utils.seed_all import seed_all
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)



class Gen_traindata:
    def __init__(self, depth_model='DepthAnythingv2', root="outputs_traindata", save_dir=None, sr=False, layerpano_dir=None):
        if save_dir is None:
            save_dir = layerpano_dir.split("/")[-1]

        self.save_dir = save_dir
        self.device = 'cuda'
        self.sr = sr
        self.layerpano_dir = layerpano_dir
        self.depth_model = depth_model
        self.pipe_dp = self.load_depth_inpainting_model()
        self.layerexp_H, self.layerexp_W = 1024, 2048
        self.traindata_dir = f'{self.save_dir}/traindata'
        self.num_layer = self.count_layers(layerpano_dir)
        print("[INFO] num layer:", self.num_layer)
        self.layerdir_list = []
        for i in range(self.num_layer + 1):
            layer_i_dir = f'{self.traindata_dir}/layer{i}'
            os.makedirs(layer_i_dir, exist_ok=True)
            self.layerdir_list.append(layer_i_dir)


    def count_layers(self, folder_path):
        subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith('layer')]        
        return len(subfolders)

    def load_depth_inpainting_model(self):
        dtype = torch.float16
        self.model_path = "checkpoints/Infusion"
        seed = int(time.time())
        seed_all(seed)


        vae = AutoencoderKL.from_pretrained(self.model_path,subfolder='vae',torch_dtype=dtype)
        scheduler = DDIMScheduler.from_pretrained(self.model_path,subfolder='scheduler',torch_dtype=dtype)
        text_encoder = CLIPTextModel.from_pretrained(self.model_path,subfolder='text_encoder',torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained(self.model_path,subfolder='tokenizer',torch_dtype=dtype)
        
        unet = UNet2DConditionModel.from_pretrained(self.model_path,subfolder="unet",
                                                    in_channels=13, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True,
                                                    torch_dtype=dtype)
        
        pipe = DepthEstimationInpaintPipeline(unet=unet,
                                       vae=vae,
                                       scheduler=scheduler,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       )

        try:

            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers

        pipe = pipe.to(self.device)
        return pipe


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
                
    def generate_traindata(self, 
                    layerpano_rgbs,        #N
                    layerpano_bagel_masks, #N
                    pano_align_masks,      #N-1
                    pano_align_masks_sharp
                    ):
        
        if self.depth_model == 'zoedepth':
            max_val = 1
            scale = 1
        else:
            max_val = 65535
            scale = 200

        ##########

        layerpano_depths = []
        N = len(layerpano_rgbs)
        pano_rgb_base = cv2.resize(layerpano_rgbs[-1], (self.layerexp_W, self.layerexp_H))                 #[h,w,3] [0-255] uint8
        pano_depth_base = self.generate_panodepth(pano_rgb_base)       #pano_depth : [1024,2048] [0-1] float64
        layerpano_depths.append(pano_depth_base)
        depthmap_utils.depth_visual_save(pano_depth_base, f"{self.traindata_dir}/pano_depth_layer{N-1}.png")


        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_color = (cmap(pano_depth_base)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(f"{self.traindata_dir}/pano_depth_layer_rgb{N-1}.png", depth_color)

        pano_depths_aligned = [(layerpano_depths[-1]* max_val).astype(np.int32)]  #[0-65535]
        
        #[layer0_rgb, layer1_rgb, layer2_rgb, layerInput]
        
        min_val = pano_depth_base.min()
        for i in range(N-1):
            id = i+1

            pano_rgb = cv2.resize(layerpano_rgbs[-(id+1)], (self.layerexp_W, self.layerexp_H)) 
            mask = pano_align_masks[-id].astype(float)
            mask[mask>0.5]=1
            mask[mask<0.5]=0
            pano_depth_base_i = layerpano_depths[-id] 

            pipe_out = self.pipe_dp(input_image=pano_rgb,
                                    depth_numpy=pano_depth_base_i,
                                    mask = mask,
                                    )

            depth_pred: np.ndarray = pipe_out.depth_np
            depth_pred = min_val + depth_pred * (1 - min_val)
            depthmap_utils.depth_visual_save(depth_pred, f"{self.traindata_dir}/pano_depth_layer{N-id-1}.png")

            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_color = (cmap(depth_pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(f"{self.traindata_dir}/pano_depth_layer_rgb{N-id-1}.png", depth_color)
            # cv2.imwrite(f"{self.traindata_dir}/pano_rgb_layer{N-id-1}.png", cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR))

            layerpano_depths.insert(0, depth_pred)

        layerpano_rgbs_unaligned = layerpano_rgbs
        layerpano_rgbs = self.rgb_alignment(layerpano_rgbs_unaligned, pano_align_masks_sharp)

        for i in range(len(layerpano_depths)):
            layerpano_depths[i] = layerpano_depths[i] * max_val
        pano_depths_aligned = layerpano_depths


        for i in range(N):
            dir_now = self.layerdir_list[i]

            pano_depths_aligned[i] = cv2.resize(pano_depths_aligned[i].astype(np.float64), (self.layer_rgb_raw_W, self.layer_rgb_raw_H)).astype(np.int32)

            pano_rgb = layerpano_rgbs[i]                #[h,w,3] [0-255] uint8


            pano_bagel_mask = cv2.resize(layerpano_bagel_masks[i], (self.layer_rgb_raw_W, self.layer_rgb_raw_H))  #[h,w,3] [0-255] uint8
            pano_depth = pano_depths_aligned[i] / max_val

            depthmap_utils.depth_visual_save(pano_depth, f"{dir_now}/pano_RGBdepth_aligned_{i}.png")

            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_color = (cmap(pano_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            depth_color_pil = Image.fromarray(depth_color)
            depth_color_pil.save(f"{dir_now}/pano_RGBdepth_aligned_light_{i}.png")

            depth_aligned = pano_depths_aligned[i]
            depth_aligned_pil = Image.fromarray(depth_aligned)
            depth_aligned_pil.save(f'{dir_now}/pano_depth_aligned_{i}.png')

            pointcloud_utils.depthmap2pointcloud_erp(pano_depth * scale , pano_rgb, f"{dir_now}/pcd_rgb_layer{i}.ply")
            pointcloud_utils.depthmap2pointcloud_erp(pano_depth * scale, pano_bagel_mask, f"{dir_now}/pcd_mask_layer{i}.ply")
            
            frame_dir = f'{dir_now}/frames'
            os.makedirs(frame_dir, exist_ok=True)

            self.gen_frames_data(pano_rgb, pano_depth * scale , frame_dir)



    def gen_frames_data(self, pano, pano_depth, save_dir, n=8):
        #pano_depth [h,w] int32 [0-65535]
        pano_H = pano.shape[0]
        pano_W = pano.shape[1]

        pers_img_size = int((pano_H /1024) * 512)
        print("frame data size:", pers_img_size)
        theta_list = []
        phi_list = []

        for i in range(n):
            th = (360/n) * i
            theta_list.append(th)
        
        theta_list = theta_list + theta_list + theta_list
        phi_list = [45]*n + [0]*n + [-45]*n

        equ = E2P.Equirectangular(pano) 
        equ_depth = E2P.Equirectangular(pano_depth.astype(np.float32)) 


        for i in range(len(theta_list)):
            th = theta_list[i]
            ph = phi_list[i]

            pose_gs = gcd_pose_gs(th, ph)

            pers_img = equ.GetPerspective(90, th, ph, pers_img_size, pers_img_size)
            pers_img = np.clip(pers_img, 0, 255).astype(np.uint8)
            Image.fromarray(pers_img).save(f'{save_dir}/rgb_{i}.png')

            # pers_depth = equ_depth.GetPerspective(90, th, ph, pers_img_size, pers_img_size)
            # # pers_depth = np.clip(pers_depth, 0, 65535).astype(np.int32)
            # pers_depth = np.clip(pers_depth, 0, 255).astype(np.uint8)

            # Image.fromarray(pers_depth).save(f'{save_dir}/depth_{i}.png')

            np.save(f'{save_dir}/transform_matrix_{i}.npy', pose_gs)

    def rgb_alignment(self, rgbs, masks):
        N = len(masks)
        rgbs_aligned = []
        rgbs_aligned.append(rgbs[0])
        h, w = rgbs[0].shape[0], rgbs[0].shape[1]
        
        for i in range(N):
            mask = np.array(masks[i])
            mask = mask / mask.max()
            # mask = mask.reshape((w, h, 1))
            mask = cv2.resize(mask, (w, h)) 
            mask = mask.reshape((h, w, 1))
            
            rgbs[i+1] = cv2.resize(rgbs[i+1], (w, h)) 
            
            aligned = (1 - mask) * rgbs_aligned[i] + mask * rgbs[i+1]
            rgbs_aligned.append(aligned)
        return rgbs_aligned

    def run(self):
        
        layerpano_dir =  self.layerpano_dir  #"/mnt/petrelfs/yangshuai/4drender/PanFusion/outputs_inpaint_city1"     #"/mnt/petrelfs/yangshuai/4drender/PanFusion/outputs_inpaint_nature1"
        #  now defined a pano divided into [layer0, layer1, layer2(input pano/ outermost layer)]


        layerInput_path = f'{layerpano_dir}/rgb_sr.png' if self.sr else f'{layerpano_dir}/rgb.png'                
        layerInput = self.readImg(layerInput_path)

        self.layer_rgb_raw_H , self.layer_rgb_raw_W = layerInput.shape[0], layerInput.shape[1]
        layer0_bagel_mask = np.ones((self.layerexp_H, self.layerexp_W, 3))

        pano_align_masks = []
        pano_align_masks_sharp = []

        layerpano_rgbs = []
        layerpano_bagel_masks = [layer0_bagel_mask]



        for i in range(self.num_layer):
            
            layer_rgb_path = f'{layerpano_dir}/layer{i}/layer{i}_inpaint_sr.png'  if self.sr else f'{layerpano_dir}/layer{i}/layer{i}_inpaint.png'         
            layer_mask_path = f'{layerpano_dir}/layer{i}/layer{i}_mask.png'  if i==self.num_layer-1 else f'{layerpano_dir}/layer{i}/layer{i}_mask_new.png'              
            layer_smoothmask_path = f'{layerpano_dir}/layer{i}/layer{i}_mask_smooth.png' if i==self.num_layer-1 else f'{layerpano_dir}/layer{i}/layer{i}_mask_smooth_new.png'

            mask_smooth = Image.open(layer_smoothmask_path).convert('L')
            mask_sharp = Image.open(layer_mask_path).convert('L')

            pano_align_masks.append(np.array(mask_smooth))
            pano_align_masks_sharp.append(np.array(mask_sharp))

            layer_rgb = self.readImg(layer_rgb_path)
            layer_next_bagel_mask = make_bagel_mask(layer_mask_path, layer_smoothmask_path)

            layerpano_rgbs.append(layer_rgb)
            layerpano_bagel_masks.append(layer_next_bagel_mask)

        layerpano_rgbs.append(layerInput)


        N = self.num_layer + 1
        
        for i in range(N):
            bagel_mask = layerpano_bagel_masks[i]
            bagel_mask= (bagel_mask * 255).astype(np.uint8)
            bagel_mask = np.repeat(bagel_mask, 3, axis=2)     #[h,w,3] [0-255] uint8
            layerpano_bagel_masks[i] = bagel_mask

        
        traindata = self.generate_traindata(layerpano_rgbs, 
                                      layerpano_bagel_masks,
                                      pano_align_masks,
                                      pano_align_masks_sharp
                                            )
        
        return traindata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for PanoScene4D')
    parser.add_argument('--depth_model', type=str, default='DepthAnythingv2') #['midas3', 'DepthAnythingv2']
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--sr', action='store_true')
    parser.add_argument('--root', type=str, default="outputs") #outputs_traindata_kernelbig_depth
    parser.add_argument('--layerpano_dir', type=str, default="")
    args = parser.parse_args()



    gen_traindata = Gen_traindata(depth_model =args.depth_model, 
                                  layerpano_dir = args.layerpano_dir,   #"/mnt/petrelfs/yangshuai/4drender/PanFusion/outputs_inpaint_city1"     #"/mnt/petrelfs/yangshuai/4drender/PanFusion/outputs_inpaint_nature1"
                                  save_dir=args.save_dir, 
                                  root = args.root,
                                  sr=args.sr)


    traindata = gen_traindata.run()
