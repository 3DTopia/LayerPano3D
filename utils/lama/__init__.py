# https://github.com/advimman/lama

import yaml
import torch
from omegaconf import OmegaConf
import numpy as np

from einops import rearrange
import os
from utils.lama.saicinpainting.training.trainers import load_checkpoint
from PIL import Image

class LamaInpainting:
    

    def __init__(self):
        self.model = None
        self.device = 'cuda'
        self.load_model()

    def load_model(self):
        modelpath = os.path.join('checkpoints', "ControlNetLama.pth")

        config_path = 'utils/lama/config.yaml'
        cfg = yaml.safe_load(open(config_path, 'rt'))
        cfg = OmegaConf.create(cfg)
        cfg.training_model.predict_only = True
        cfg.visualizer.kind = 'noop'
        self.model = load_checkpoint(cfg, os.path.abspath(modelpath), strict=False, map_location='cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, color, mask):
        if self.model is None:
            self.load_model()
        self.model.to(self.device)
        
        color = np.ascontiguousarray(np.array(color)).astype(np.float32) / 255.0
        mask = np.ascontiguousarray(np.array(mask)).astype(np.float32) / 255.0
        
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        
        with torch.no_grad():
            color = torch.from_numpy(color).float().to(self.device)
            mask = torch.from_numpy(mask).float().to(self.device)
            mask = (mask > 0.5).float()
            color = color * (1 - mask)
            image_feed = torch.cat([color, mask], dim=2)
            image_feed = rearrange(image_feed, 'h w c -> 1 c h w')
            result = self.model(image_feed)[0]
            result = rearrange(result, 'c h w -> h w c')
            result = result * mask + color * (1 - mask)
            result *= 255.0
            return result.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
