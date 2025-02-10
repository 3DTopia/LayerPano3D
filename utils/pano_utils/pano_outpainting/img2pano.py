# Copyright © Alibaba, Inc. and its affiliates.
import random
from typing import Any, Dict
import torchvision
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from diffusers import (ControlNetModel, DiffusionPipeline,
                       EulerAncestralDiscreteScheduler,
                       UniPCMultistepScheduler)
from PIL import Image
from realesrgan import RealESRGANer

from .pipeline_i2p import StableDiffusionImage2PanoPipeline


class Img2Pano(DiffusionPipeline):
    def __init__(self, model: str ='load', device: str = 'cuda', **kwargs):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'
                              ) if device is None else device
        if device == 'gpu':
            device = 'cuda'

        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        enable_xformers_memory_efficient_attention = kwargs.get(
            'enable_xformers_memory_efficient_attention', True)

        model_id = model + '/sr-base/'

        # init i2p model
        controlnet = ControlNetModel.from_pretrained(model + '/sd-i2p', torch_dtype=torch.float16)

        self.pipe = StableDiffusionImage2PanoPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch_dtype).to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)
        # remove following line if xformers is not installed
        try:
            if enable_xformers_memory_efficient_attention:
                self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
        self.pipe.enable_model_cpu_offload()

        # init realesrgan model
        sr_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2)
        netscale = 2

        model_path = model + '/RealESRGAN_x2plus.pth'

        dni_weight = None
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=sr_model,
            tile=384,
            tile_pad=20,
            pre_pad=20,
            half=False,
            device=device,
        )


    @staticmethod
    def blend_h(a, b, blend_extent):
        blend_extent = min(a.shape[1], b.shape[1], blend_extent)
        for x in range(blend_extent):
            b[:, x, :] = a[:, -blend_extent
                           + x, :] * (1 - x / blend_extent) + b[:, x, :] * (
                               x / blend_extent)
        return b

    def __call__(self,
                 prompt=None,
                 negative_prompt=None,
                 control_image=None,
                 num_inference_steps=20,
                 guidance_scale=7.0,
                 seed = 25,
                 upscale = True,
                 ) -> Dict[str, Any]:

        preset_a_prompt = 'photorealistic, trend on artstation, ((best quality)), ((ultra high res))'
        preset_n_prompt = 'persons, complex texture, small objects, sheltered, blur, worst quality, '\
                          'low quality, zombie, logo, text, watermark, username, monochrome, '\
                          'complex lighting'
        if negative_prompt is None:
            negative_prompt = preset_n_prompt
                
        # print(f'Test with prompt: {prompt}')

        if seed == -1:
            seed = random.randint(0, 65535)
        print(f'global seed: {seed}')

        generator = torch.manual_seed(seed)

        prompt = '<360panorama>, ' + prompt + ', ' + preset_a_prompt
        output_img = self.pipe(
            prompt,
            image=(control_image[:, 1:, :, :] / 0.5 - 1.0),
            control_image=control_image,
            controlnet_conditioning_scale=1.0,
            strength=1.0,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            height=512,
            width=1024,
            guidance_scale=guidance_scale,
            generator=generator).images[0]

        if not upscale:
            print('finished')
        else:
        
            print('running upscaler step. Super-resolution with Real-ESRGAN')
            w = output_img.size[0]
            blend_extend = 10
            outscale = 2
            output_img = np.array(output_img)
            output_img = np.concatenate(
                [output_img, output_img[:, :blend_extend, :]], axis=1)
            output_img, _ = self.upsampler.enhance(
                output_img, outscale=outscale)
            output_img = self.blend_h(output_img, output_img,
                                      blend_extend * outscale)
            
            output_img = Image.fromarray(output_img[:, :w * outscale, :])
            print('finished')

        return output_img
