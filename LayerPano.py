
import os
import datetime
import warnings
from random import randint
from loguru import logger
warnings.filterwarnings(action='ignore')
import imageio

import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from plyfile import PlyData
from torchvision.transforms import ToPILImage, ToTensor



import utils.pano_utils.Equirec2Perspec as E2P
import utils.pano_utils.multi_Perspec2Equirec as m_P2E


from arguments import GSParams, CameraParams, ModelHiddenParams
from gaussian_renderer import render
from scene import Scene, GaussianModel, LayerGaussian

from utils.loss import l1_loss, ssim, lpips_loss
from utils.camera import load_json

from utils.depth_utils import colorize
from utils.image import psnr
from utils.paint_utils import functbl
from utils.trajectory import get_pcdGenPoses
from scene.cameras import MiniCam2


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


get_kernel = lambda p: torch.ones(1, 1, p * 2 + 1, p * 2 + 1).to('cuda')
t2np = lambda x: (x[0].permute(1, 2, 0).clamp_(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
np2t = lambda x: (torch.as_tensor(x).to(torch.float32).permute(2, 0, 1) / 255.0)[None, ...].to('cuda')
pad_mask = lambda x, padamount=1: t2np(
    F.conv2d(np2t(x[..., None]), get_kernel(padamount), padding=padamount))[..., 0].astype(bool)



def check_cuda_memo(info="", device=0):
    print(f"================= cuda memory info {info} ==================")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    
    free_memory = total_memory - (allocated_memory + cached_memory)
    
    print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
    print(f"Cached memory: {cached_memory / 1024**2:.2f} MB")
    print(f"Free memory: {free_memory / 1024**2:.2f} MB")
    print(f"=====================================================\n")
    



class LayerPano:
    def __init__(self, save_dir=None): 
        self.init_logger()
        self.save_dir = save_dir
        self.opt = GSParams()
        self.cam = CameraParams()
        self.hyper = ModelHiddenParams()
        self.device = 'cuda'
        self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        
        bg_color = [1, 1, 1]  #[0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
        self.step=0
        self.is_upper_mask_aggressive = True
        
        self.data_path = os.path.join(self.save_dir, 'data')
        self.pers_path = os.path.join(self.data_path, 'perspective_imgs')        

        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.pers_path, exist_ok=True)
            
    def save_img(self, x, path):
        if np.max(x) > 1:
            x = x.astype(np.uint8)
        else:
            x = (x*255).astype(np.uint8)
        image = Image.fromarray(x)
        image.save(path)
        
    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)

    def count_layer(self, base_dir):
        count = 0
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and item.startswith("layer"):
                count += 1

        return count

    def readImg(self, path):
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        return img


    def create(self, input_dir, outlier_thresh,):

        input_dir = os.path.join(input_dir,'traindata')
        n_layer = self.count_layer(input_dir)
        # n_layer = 1
        print('Layers of Pano:', n_layer)
        self.outlier_thresh = outlier_thresh
        print('Outlier Thresh', self.outlier_thresh)
        
        gaussians_prev = None
        for layer_idx in range(n_layer):
            self.traindata = self.load_pcd_and_perspectives(input_dir, layer_idx)
            if layer_idx == 0:
                n_iterations = 3001
            else:
                n_iterations = 2001

            self.gaussians = LayerGaussian(self.opt.sh_degree, outlier_thresh=self.outlier_thresh)
            self.scene = Scene(self.traindata, gaussians_prev, self.gaussians, self.opt)        
                        
            self.training(layer_idx, n_iterations)

            
            
            
            self.timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            gaussians_prev = self.gaussians.wrap_gaussian()
            outfile = self.save_ply(os.path.join(self.save_dir, f'gsplat_layer{layer_idx}.ply'))

        return outfile
    
    def save_ply(self, fpath=None, type='3D'):
            
        if type == '3D':
            self.gaussians.save_ply(fpath)
        else:
            if not os.path.exists(fpath):
                self.gaussians_4d.save_ply(fpath)
            else:
                self.gaussians_4d.load_ply(fpath)
        return fpath


    def render_video(self, preset, phi=0):
        
        if preset == '360':
            preset='pers2pano'
            poses, theta_list, phi_list = get_pcdGenPoses(preset, {'n_views': 80, 'phi': phi})
        else:
            poses = get_pcdGenPoses(preset)

        
        videopath = os.path.join(self.save_dir, 'results', f'{preset}_v{phi}.mp4')
        depthpath = os.path.join(self.save_dir, 'results', f'depth_{preset}_v{phi}.mp4')
        
        views = []

        
        for i in range(len(poses)):
            pose = poses[i]
            cur_cam = MiniCam2(pose, self.cam.W, self.cam.H, self.cam.fovx, self.cam.fovy)
            views.append(cur_cam)
            
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8


        iterable_render = views

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
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


        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=10, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=10, quality=8)
        return videopath, depthpath

    def render_video(self, preset, phi=0):
        
        if preset == '360':
            preset='pers2pano'
            poses, theta_list, phi_list = get_pcdGenPoses(preset, {'n_views': 80, 'phi': phi})
        else:
            poses = get_pcdGenPoses(preset)

        
        videopath = os.path.join(self.save_dir, 'results', f'{preset}_v{phi}.mp4')
        depthpath = os.path.join(self.save_dir, 'results', f'depth_{preset}_v{phi}.mp4')
        

        views = []

        
        for i in range(len(poses)):
            pose = poses[i]
            cur_cam = MiniCam2(pose, self.cam.W, self.cam.H, self.cam.fovx, self.cam.fovy)
            views.append(cur_cam)
            
        framelist = []
        depthlist = []
        dmin, dmax = 1e8, -1e8


        iterable_render = views

        for view in iterable_render:
            results = render(view, self.gaussians, self.opt, self.background)
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


        # depthlist = [colorize(depth, vmin=dmin, vmax=dmax) for depth in depthlist]
        depthlist = [colorize(depth) for depth in depthlist]
        if not os.path.exists(videopath):
            imageio.mimwrite(videopath, framelist, fps=10, quality=8)
        if not os.path.exists(depthpath):
            imageio.mimwrite(depthpath, depthlist, fps=10, quality=8)
        return videopath, depthpath

    def training(self, layer_idx, n_iterations):
        
        if not self.scene:
            raise('Build 3D Scene First!')
        
        self.opt.iterations = n_iterations

        iterable_gauss = range(1, self.opt.iterations + 1)

        # iterable_gauss = range(1, n_iterations + 1)
        tb_writer = self.prepare_logger()
        progress_bar = tqdm(range(0, n_iterations), desc="Training progress")
        ema_loss_for_log = 0.0

        # iter_start = torch.cuda.Event(enable_timing = True)
        # iter_end = torch.cuda.Event(enable_timing = True)

        for iteration in tqdm(iterable_gauss):
            self.gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.gaussians.oneupSHdegree()

            # Pick a random Camera
            viewpoint_stack = self.scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            # import pdb; pdb.set_trace()
            # Render
            render_pkg = render(viewpoint_cam, self.gaussians, self.opt, self.background)
            
            image, mask, depth = render_pkg['render'], render_pkg['mask'], render_pkg['depth'] #[c,h,w]


            viewspace_point_tensor, visibility_filter, radii = render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii']
 

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            

            Ll1 = l1_loss(image, gt_image)
            if iteration % 1000 == 1:
                print('l1 loss:', Ll1)

            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image)) #+ 0.5 * Ldepth
            loss.backward()

            with torch.no_grad():
                # Densification
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        
                if iteration % 10 == 0:
                    loss_dict = {
                        "Loss": f"{ema_loss_for_log:.{5}f}",
                        "Points": f"{len(self.gaussians.get_xyz)}"
                    }
                    progress_bar.set_postfix(loss_dict)

                    progress_bar.update(10)

                if iteration == n_iterations:
                    progress_bar.close()

                if iteration < self.opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    self.gaussians.update_identity_mask()
                    visibility_filter_part = radii[self.gaussians.identity_mask] > 0
                    self.gaussians.max_radii2D[visibility_filter_part] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_part], radii[self.gaussians.identity_mask][visibility_filter_part])
                    self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter_part, self.gaussians.identity_mask)

                    if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                        size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                        self.gaussians.densify_and_prune(
                            self.opt.densify_grad_threshold, 0.01, self.scene.cameras_extent, size_threshold)
                    
                    if (iteration % self.opt.opacity_reset_interval == 0 
                        or (self.opt.white_background and iteration == self.opt.densify_from_iter)
                    ):
                        self.gaussians.reset_opacity()

                # Optimizer step
                if iteration < self.opt.iterations:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none = True)
        
        
        # self.compose_pano()

    

    def compose_pano(self):
        to_pil = ToPILImage()
        phi_all = [0, -45, 45, -80, 80]
        cam_fov90 = CameraParams(fov=90)

        pers_img = []
        F_T_P = []        
        
        for phi in phi_all:

            pers_img_tmp = []
            F_T_P_tmp = []   

            persdata, theta_list, phi_list = get_pcdGenPoses("pers2pano",{'n_views': 10, 'phi': phi })    
            n_pers = len(persdata)
            
            path = os.path.join(self.pers_path, f'pers_phi{phi}');os.makedirs(path, exist_ok=True)
            
            for i in range(n_pers):
                pose = persdata[i]
                cur_cam = MiniCam2(pose, cam_fov90.W, cam_fov90.H, cam_fov90.fovx, cam_fov90.fovy)
                render_pkg = render(cur_cam, self.gaussians, self.opt, self.background, render_only=True)
                #image:[3,H,W]
                image = render_pkg['render']   # depth[1, 512, 512]
                
                image = to_pil(image.cpu()); image.save(os.path.join(path, f'pers_{i}.jpg'))
                image = np.array(image)
                
                pers_img.append(image)
                F_T_P.append([cam_fov90.fov_deg, theta_list[i], phi_list[i]])

                pers_img_tmp.append(image)
                F_T_P_tmp.append([cam_fov90.fov_deg, theta_list[i], phi_list[i]])

            ee = m_P2E.Perspective(pers_img, F_T_P)
            pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)            
            self.save_img(pano_img, os.path.join(path, f'pano_{phi}.jpg'))


            ee = m_P2E.Perspective(pers_img_tmp, F_T_P_tmp)
            pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)            
            self.save_img(pano_img, os.path.join(path, f'pano_tmp_{phi}.jpg'))

        ee = m_P2E.Perspective(pers_img, F_T_P)
        pano_img, pano_mask = ee.GetEquirec(1024, 2048, return_mask=True)
        
        self.save_img(pano_img, os.path.join(self.save_dir, f'pano.jpg'))
        self.save_img(pano_mask, os.path.join(self.save_dir, f'pano_mask.jpg'))

    def getmask(self, img):
        # img [h,w,3]
        mask = np.sum(img, axis=-1)
        mask = np.array((mask > 0)).astype(np.float32)
        return mask

    def pano2pers(self, pano, viewangle, N, time=None, name=None):
        pers_img=[]
        if not name:
            name = 'pers_split'
        equ = E2P.Equirectangular(pano)
        for i in range(N):
            theta = 360 - (viewangle/N)*i
            img = equ.GetPerspective(self.cam.fov_deg, theta, 0, self.cam.H, self.cam.W)
            
            img = np.clip(img, 0, 255).astype(np.uint8)
            if time:
                pil_img = Image.fromarray(img); pil_img.save(os.path.join(self.save_dir, f'{name}{time}_{i}.jpg'))
            else:
                pil_img = Image.fromarray(img); pil_img.save(os.path.join(self.save_dir, f'{name}_{i}.jpg'))
            pers_img.append(np.array(pil_img))
        return pers_img
 

    
    def load_pcd_and_perspectives(self, parent_dir, idx):
        load_dir = os.path.join(parent_dir, f'layer{idx}')
        pcd_points, pcd_colors = self.load_pcd(os.path.join(load_dir, f'pcd_rgb_layer{idx}.ply'))
        _, pcd_masks = self.load_pcd(os.path.join(load_dir, f'pcd_mask_layer{idx}.ply'))
        pcd_colors = pcd_colors / pcd_colors.max()
        pcd_masks = pcd_masks / pcd_masks.max()

        assert pcd_points.shape[0] == pcd_masks.shape[0]
        if len(pcd_points) > 3000000:
            ratio = len(pcd_points) // 3000000 + 1
            print('Warning: PointCloud is too large {}, downsampling by ratio of {}'.format(len(pcd_points),ratio))
            pcd_points = pcd_points[::ratio]
            pcd_colors = pcd_colors[::ratio]
            pcd_masks = pcd_masks[::ratio]

        print('[INFO] !!! Loaded {} points from Layer {}.'.format(pcd_points.shape, idx))
        num_frames = 24
        frames = []
        for frame_idx in range(num_frames):
            pers_rgb = Image.open(os.path.join(load_dir, f'frames/rgb_{frame_idx}.png'))
            pose_gs = np.load(os.path.join(load_dir, f'frames/transform_matrix_{frame_idx}.npy'))
            frames.append({'image': pers_rgb, 'transform_matrix': pose_gs})
        
        W, H = frames[-1]['image'].size

        self.cam.W = W
        self.cam.H = H
        self.cam.fovx = math.radians(90)
        self.cam.fovy = self.cam.H * self.cam.fovx / self.cam.W

        self.cam.fov = (self.cam.fovx, self.cam.fovy)
        self.cam.fov_deg = 90

        return {
            'fov': self.cam.fov_deg,
            'W': self.cam.W,
            'H': self.cam.H,
            'pcd_points': pcd_points,
            'pcd_colors': pcd_colors,
            'pcd_masks': pcd_masks,
            'frames': frames
            }
    
    def load_pcd(self, pcd_path):
        plydata = PlyData.read(pcd_path)
        vertices = plydata['vertex']
        x, y, z = vertices['x'], vertices['y'], vertices['z']
        r, g, b = vertices['red'], vertices['green'], vertices['blue']
        points = np.stack([x,y,z], axis=-1)
        colors = np.stack([r,g,b], axis=-1)
        return points, colors

    def prepare_logger(self):    
        # with open(os.path.join(self.save_dir, "cfg_args"), 'w') as cfg_log_f:
        #     cfg_log_f.write(str(Namespace(**vars(self.opt))))
        # Create Tensorboard writer
        tb_writer = None
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(self.save_dir)
        else:
            print("Tensorboard not available: not logging progress")
        return tb_writer
