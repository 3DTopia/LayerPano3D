import os
import sys
import cv2
import glob
import argparse
import open_clip
import numpy as np
from PIL import Image
import safetensors.torch

import torch
from torchvision import transforms
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, PNDMScheduler, LCMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler#, StableDiffusionControlNetPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

# sys.path.append('pasd')

from pasd.pipelines.pipeline_pasd import StableDiffusionControlNetPipeline
from pasd.myutils.misc import load_dreambooth_lora
from pasd.myutils.wavelet_color_fix import wavelet_color_fix
#from annotator.retinaface import RetinaFaceDetection


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def load_pasd_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    if args.use_pasd_light:
        from pasd.models.pasd_light.unet_2d_condition import UNet2DConditionModel
        from pasd.models.pasd_light.controlnet import ControlNetModel
    else:
        from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel
        from pasd.models.pasd.controlnet import ControlNetModel
    # Load scheduler, tokenizer and models.
    if args.control_type=="grayscale":
        scheduler = UniPCMultistepScheduler.from_pretrained("/".join(args.pasd_model_path.split("/")[:-1]), subfolder="scheduler")
    else:
        scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.pasd_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.pasd_model_path, subfolder="controlnet")

    personalized_model_root = "checkpoints/personalized_models"
    if args.use_personalized_model and args.personalized_model_path is not None:
        if os.path.isfile(f"{personalized_model_root}/{args.personalized_model_path}"):
            unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, f"{personalized_model_root}/{args.personalized_model_path}", 
                                                           blending_alpha=args.blending_alpha, multiplier=args.multiplier)
        else:
            unet = UNet2DConditionModel.from_pretrained_orig(personalized_model_root, subfolder=f"{args.personalized_model_path}") # unet_disney

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    #validation_pipeline.enable_vae_tiling()
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.encoder_tiled_size, decoder_tile_size=args.decoder_tiled_size)

    if args.use_lcm_lora:
        # load and fuse lcm lora
        validation_pipeline.load_lora_weights(args.lcm_lora_path)
        validation_pipeline.fuse_lora()
        validation_pipeline.scheduler = LCMScheduler.from_config(validation_pipeline.scheduler.config)

    return validation_pipeline

def load_high_level_net(args, device='cuda'):
    if args.high_level_info == "classification":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        resnet = resnet50(weights=weights)
        resnet.eval()
        return resnet, preprocess, weights.meta["categories"]
    elif args.high_level_info == "detection":
        from annotator.yolo import YoLoDetection
        yolo = YoLoDetection()
        return yolo, None, None
    elif args.high_level_info == "caption":
        if args.use_blip:
            from lavis.models import load_model_and_preprocess
            model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
            return model, vis_processors, None
        else:
            #import open_clip
            model, _, transform = open_clip.create_model_and_transforms(
                model_name="coca_ViT-L-14",
                pretrained="mscoco_finetuned_laion2B-s13B-b90k"
                )
            return model, transform, None
    else:
        return None, None, None
    
def get_validation_prompt(args, image, model, preprocess, category, device='cuda'):
    validation_prompt = ""

    if args.high_level_info == "classification":
        batch = preprocess(image).unsqueeze(0)
        prediction = model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = category[class_id]
        #print(f"{category_name}: {100 * score:.1f}%")
        if score >= 0.1:
            validation_prompt = f"{category_name}, " if args.prompt=="" else f"{args.prompt}, {category_name}, "
    elif args.high_level_info == "detection":
        clses, confs, names = model.detect(image)
        #print(cls, conf, names)
        count = {}
        for cls, conf in zip(clses, confs):
            name = names[cls]
            if name in count: 
                count[name] += 1
            else:
                count[name] = 1
        for name in count:
            validation_prompt += f"{count[name]} {name}, "
        validation_prompt = validation_prompt if args.prompt=="" else f"{args.prompt}, {validation_prompt}"
    elif args.high_level_info == "caption":
        if args.use_blip:
            image = preprocess["eval"](image).unsqueeze(0).to(device)
            caption = model.generate({"image": image}, num_captions=1)[0]
            caption = caption.replace("blurry", "clear").replace("noisy", "clean") #
            validation_prompt = caption if args.prompt=="" else f"{caption}, {args.prompt}"
        else:
            image = preprocess(image).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                generated = model.generate(image)
            caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
            caption = caption.replace("blurry", "clear").replace("noisy", "clean") #
            validation_prompt = caption if args.prompt=="" else f"{caption} {args.prompt}"
    else:
        validation_prompt = "" if args.prompt=="" else f"{args.prompt}, "
    
    return validation_prompt




def count_layer(base_dir):
    count = 0
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("layer"):
            count += 1

    return count


def main(args, enable_xformers_memory_efficient_attention=True,):
    
    
    num_layer  = count_layer(args.inputs_dir)
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("PASD")

    pipeline = load_pasd_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model, preprocess, category = load_high_level_net(args, accelerator.device)

    resize_preproc = transforms.Compose([
        transforms.Resize(args.process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ] if args.control_type=="realisr" else [
        transforms.Resize(args.process_size, max_size=args.process_size*2, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
                
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        # if os.path.isdir(args.image_path):
        #     image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
        # else:
        #     image_names = [args.image_path]
        base_dir = args.inputs_dir
        print("=============================================")
        print(f"base_dir:{base_dir}")
        image_names = []
        for i in range(num_layer):
            

            image_path = f"{base_dir}/layer{i}/layer{i}_inpaint.png"
            image_names.append(image_path)
        
        image_names.append(f"{base_dir}/rgb.png")

        index = 0
        for image_name in image_names[:]:
            
            validation_image = Image.open(image_name).convert("RGB")
            #validation_image = Image.new(mode='RGB', size=validation_image.size, color=(0,0,0))
            if args.control_type == "realisr":
                validation_prompt = get_validation_prompt(args, validation_image, model, preprocess, category)
                validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
                negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            elif args.control_type == "grayscale":
                validation_image = validation_image.convert("L").convert("RGB")
                orig_img = validation_image.copy()
                validation_prompt = get_validation_prompt(args, validation_image, model, preprocess, category, accelerator.device)
                validation_prompt = validation_prompt.replace("black and white", "color")
                negative_prompt = "b&w, color bleeding"
            else:
                raise NotImplementedError
            
            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale if args.control_type=="realisr" else 1

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))

            if min(validation_image.size) < args.process_size or args.control_type=="grayscale":
                validation_image = resize_preproc(validation_image)

            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            #width, height = validation_image.size
            resize_flag = True #

            image = pipeline(
                    args, validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, #height=height, width=width,
                    guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                ).images[0]

            # try:
            #     print(3333333333)
            #     image = pipeline(
            #             args, validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, #height=height, width=width,
            #             guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
            #         ).images[0]
            # except Exception as e:
            #     print(44444444)
            #     print(e)
            #     continue

            if args.control_type=="realisr": 
                if True: #args.conditioning_scale < 1.0:
                    image = wavelet_color_fix(image, validation_image)

                if resize_flag: 
                    image = image.resize((ori_width*rscale, ori_height*rscale))

            name, ext = os.path.splitext(os.path.basename(image_name))
            if args.control_type=='grayscale':
                
                np_image = np.asarray(image)[:,:,::-1]
                color_np = cv2.resize(np_image, orig_img.size)
                orig_np = np.asarray(orig_img)
                color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
                orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
                hires = np.copy(orig_yuv)
                hires[:, :, 1:3] = color_yuv[:, :, 1:3]
                np_image = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
                if index == num_layer:
                    output_path = f"{base_dir}/rgb_grayscale_sr.png"
                else:
                    output_path = f"{base_dir}/layer{index}/layer{index}_inpaint_grayscale_sr.png"
                cv2.imwrite(output_path, np_image)

                # cv2.imwrite(f'{args.output_dir}/{name}.png', np_image)
            else:
                # image.save(f'{args.output_dir}/{name}_sr.png')
                if index == num_layer:
                    output_path = f"{base_dir}/rgb_sr.png"
                else:
                    output_path = f"{base_dir}/layer{index}/layer{index}_inpaint_sr.png"
                print("============output_path===========",output_path)

                image.save(output_path)

            index = index + 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="checkpoints/pasd/stable-diffusion-v1-5", help="path of base SD model")
    parser.add_argument("--lcm_lora_path", type=str, default="checkpoints/pasd/lcm-lora-sdv1-5", help="path of LCM lora model")
    parser.add_argument("--pasd_model_path", type=str, default="checkpoints/pasd/checkpoint-100000", help="path of PASD model")
    parser.add_argument("--personalized_model_path", type=str, default="majicmixRealistic_v7.safetensors", help="name of personalized dreambooth model, path is 'checkpoints/personalized_models'") # toonyou_beta3.safetensors, majicmixRealistic_v6.safetensors, unet_disney
    parser.add_argument("--control_type", choices=['realisr', 'grayscale'], nargs='?', default="realisr", help="task name")
    parser.add_argument('--high_level_info', choices=['classification', 'detection', 'caption'], nargs='?', default='caption', help="high level information for prompt generation")
    parser.add_argument("--prompt", type=str, default="", help="prompt for image generation")
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k", help="additional prompt")
    parser.add_argument("--negative_prompt", type=str, default="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed", help="negative prompt")

    parser.add_argument("--inputs_dir", type=str, default="outputs_nature_inpaint", help="layerdata dir")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="mixed precision mode") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=9.0, help="classifier-free guidance scale")
    parser.add_argument("--conditioning_scale", type=float, default=1.0, help="conditioning scale for controlnet")
    parser.add_argument("--blending_alpha", type=float, default=1.0, help="blending alpha for personalized model")
    parser.add_argument("--multiplier", type=float, default=0.6, help="multiplier for personalized lora model")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="denoising steps")
    parser.add_argument("--process_size", type=int, default=768, help="minimal input size for processing") # 512?
    parser.add_argument("--decoder_tiled_size", type=int, default=224, help="decoder tile size for saving GPU memory") # for 24G
    parser.add_argument("--encoder_tiled_size", type=int, default=1024, help="encoder tile size for saving GPU memory") # for 24G
    parser.add_argument("--latent_tiled_size", type=int, default=320, help="unet latent tile size for saving GPU memory") # for 24G
    parser.add_argument("--latent_tiled_overlap", type=int, default=8, help="unet lantent overlap size for saving GPU memory") # for 24G
    parser.add_argument("--upscale", type=int, default=1, help="upsampling scale")
    parser.add_argument("--use_personalized_model", action="store_true", help="use personalized model or not")
    parser.add_argument("--use_pasd_light", action="store_true", help="use pasd or pasd_light")
    parser.add_argument("--use_lcm_lora", action="store_true", help="use lcm-lora or not")
    parser.add_argument("--use_blip", action="store_true", help="use lcm-lora or not")
    parser.add_argument("--init_latent_with_noise", action="store_true", help="initial latent with pure noise or not")
    parser.add_argument("--added_noise_level", type=int, default=900, help="additional noise level")
    parser.add_argument("--offset_noise_scale", type=float, default=0.0, help="offset noise scale, not used")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    args = parser.parse_args()
    main(args)
