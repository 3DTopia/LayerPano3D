import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import random
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from src.pipeline_flux_fill import FluxFillPipeline
from utils.lama import LamaInpainting

from transformers import AutoProcessor, LlavaForConditionalGeneration



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.6,
            crop_n_layers=0,
            crop_n_points_downscale_factor=0,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )


model_id = "llava-hf/llava-1.5-7b-hf"
llm_model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

def generate_caption(model, raw_image):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "you are a powerful image captioner. Instead of describing the imaginary content, only describing the content one can determine confidently from the image. Do not describe the contents by itemizing them in list form. Keep it short and simple.Minimize aesthetic descriptions as much as possible. Beside, Start with The image captures a xxx"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    caption = processor.decode(output[0][2:], skip_special_tokens=True)
    caption = caption[355:]
    caption = caption.replace("The image captures ", "")
    
    return caption


def draw_mask_overlay(mask, image):
    rand_color = np.array([random.randint(0, 255) for _ in range(3)])
    colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    colored_mask = colored_mask * rand_color
    colored_mask = colored_mask.astype(np.uint8)

    # Blend the image and the colored mask
    alpha = 0.6  # Transparency factor
    overlay = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)
    return overlay



def extend_mask(image, mask, sam_masks, cumulative_mask, base_dir, n_layer):
    mask2 = np.zeros_like(mask)
    mask = mask.astype(bool)
    for i, mask_item in tqdm(enumerate(sam_masks)):
        sam_mask = mask_item['segmentation']
        union = np.logical_or(mask, sam_mask)
        intersection = np.logical_and(mask, sam_mask)
        iou = intersection.sum() / union.sum()
        iou2 = intersection.sum() / mask.sum()
        intersection = np.logical_and(mask, sam_mask)
        cumu_iou = intersection.sum() / sam_mask.sum()
        if cumu_iou > 0.35 and cumu_iou < 0.95 and iou2 < 0.9: #iou > 0.2: #or 
            mask2 = np.logical_or(mask2, sam_mask)

        sam_mask_img = draw_mask_overlay(sam_mask.astype(bool), image)
        sam_mask_img = Image.fromarray(sam_mask_img.astype(np.uint8))
        
    return mask2

def count_layer(base_dir):
    count = 0
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("layer"):
            count += 1

    return count

def get_smooth_mask(general_mask, ksize = 50):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    mask_array = cv2.dilate(general_mask.astype(np.uint8), kernel)              #[1024, 2048] uint8 1
    mask_array = (mask_array>0).astype(np.uint8)
    return mask_array

print("=============  now start generate layer panorama data ===============")

#############################################################
parser = ArgumentParser(description="Layer Decomposition Parameters")
parser.add_argument('--lora_path', type=str, default='checkpoints/pano_lora_720*1440_v1.safetensors', help='lora checkpoint path')
parser.add_argument("--base_dir", default="outputs", type=str)
parser.add_argument('--seed', type=int, default=25, help='seed')
parser.add_argument('--strength', type=float, default=0.6, help='strength')
parser.add_argument('--maxsize', type=int, default=2048, help='maxsize')
args = parser.parse_args()

base_dir = args.base_dir
num_layer  = count_layer(args.base_dir)#[layer2, layer1, layer0]

size = (int(args.maxsize // 2), args.maxsize)




if args.seed == -1:
    import random
    seed = random.randint(1, 65535)
else:
    seed = args.seed

pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights(args.lora_path) # change this.
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

lama_model = LamaInpainting()



num_layer = sum(1 for item in Path(base_dir).iterdir() if item.is_dir() and item.name.startswith("layer"))
print(f"{base_dir}-num_layer{num_layer}")
for i in range(num_layer):
    n_layer = num_layer - 1 - i
    layer_dir = f"{base_dir}/layer{n_layer}"
    
    if i != 0:
        images_dir = "{}/layer{}/layer{}_inpaint.png".format(base_dir, n_layer+1, n_layer+1)
    else:   
        images_dir = "{}/rgb.png".format(base_dir)

    image = Image.open(images_dir).convert('RGB').resize((size[1], size[0]))
    mask_sharp = Image.open(os.path.join(base_dir, 'layer{}/layer{}_mask.png'.format(n_layer,n_layer))).convert('L').resize((size[1], size[0]))
        
    mask_smooth = get_smooth_mask(np.asarray(mask_sharp))

    mask_image = Image.fromarray(mask_smooth)
    Image.fromarray(mask_smooth*255).save(os.path.join(base_dir, 'layer{}/layer{}_mask_smooth.png'.format(n_layer,n_layer)))
    w,h = image.size

    mask = np.array(mask_image)
    mask = mask / mask.max()
    mask = mask.astype(bool)
    image_array = np.asarray(image)
    if i != 0:
        sam_masks = mask_generator.generate(image_array)
        mask2 = extend_mask(image_array, np.array(mask_sharp) / np.array(mask_sharp).max(), sam_masks, cumulative_mask, base_dir, n_layer)
        mask2_smooth = get_smooth_mask(mask2)
        mask = np.logical_or(mask, mask2_smooth)
        mask_sharp = np.logical_or(mask_sharp, mask2)
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_sharp_img = Image.fromarray(mask_sharp.astype(np.uint8) * 255)

        color_masked_img = draw_mask_overlay(mask_sharp.astype(bool), image_array)
        color_masked_img = Image.fromarray(color_masked_img.astype(np.uint8))
        color_masked_img.save(os.path.join(base_dir, 'layer{}/layer{}_maskedimg_color.png'.format(n_layer,n_layer)))
        
        mask_sharp_img.save(os.path.join(base_dir, 'layer{}/layer{}_mask_new.png'.format(n_layer,n_layer)))
        mask_image.save(os.path.join(base_dir, 'layer{}/layer{}_mask_smooth_new.png'.format(n_layer,n_layer)))
        cumulative_mask = np.logical_or(cumulative_mask, mask)

        pano_path = f"{base_dir}/layer{n_layer+1}/layer{n_layer+1}_inpaint.png"
        mask_path = f"{base_dir}/layer{n_layer}/layer{n_layer}_mask_smooth_new.png"

    else:
        cumulative_mask = mask.copy()
        pano_path = f"{base_dir}/rgb.png"
        mask_path = f"{base_dir}/layer{n_layer}/layer{n_layer}_mask_smooth.png"

    pano = cv2.imread(pano_path)
    pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)    #[1024, 2048]


    pano = cv2.resize(pano, (size[1], size[0]))
    pano_mask_pil = Image.open(mask_path).resize((size[1], size[0]))
    pano_mask = np.array(pano_mask_pil)

    
    pano_mask_lama = pano_mask / 255
    pano_mask_lama = pano_mask_lama[:, :, None].astype(np.uint8)
    
    
    tmp_size = (256,512)
    pano_lama = lama_model(cv2.resize(pano, (tmp_size[1], tmp_size[0])), cv2.resize(pano_mask, (tmp_size[1], tmp_size[0]))) #(1024, 2048, 3) uint8 255
    pano_lama = pano_lama.squeeze().astype(np.uint8)

    pano = cv2.resize(pano_lama, (size[1], size[0])) * pano_mask_lama + (1-pano_mask_lama) * pano
    
    pano_lama_pil = Image.fromarray(pano)
    pano_lama_pil.save(f'{layer_dir}/layer{n_layer}_lama.png')
    pano_pil = Image.fromarray(pano)

    prompt = generate_caption(llm_model, pano_lama_pil)

    # prompt = "Empty Scene, Nothing."
    print(f"now layer{n_layer} prompt:{prompt}")
    num_inference_steps=50
    pano_pred_pil = pipe(
                prompt=prompt,
                image=pano_lama_pil,
                mask_image=pano_mask_pil,
                height=1024,
                width=2048,
                strength = args.strength,
                guidance_scale=30,
                num_inference_steps=num_inference_steps,
                max_sequence_length=512,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]


    pano_pred_pil.save(f'{layer_dir}/layer{n_layer}_inpaint.png')   
    print(f"[INFO]:===============Layer{n_layer} DONE!!!================")
