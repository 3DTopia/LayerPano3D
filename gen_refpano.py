import torch
import os
import random
import argparse
from src.pipeline_flux import FluxPipeline



parser = argparse.ArgumentParser(description='Arguments for PanoScene4D')
parser.add_argument('--save_dir', type=str, default="outputs")
parser.add_argument('--prompt', type=str, default="Sandy beach, large driftwood in the foreground, calm sea beyond, realism style.")
parser.add_argument('--lora_path', type=str, default='checkpoints/lora.safetensors') 
parser.add_argument('--seed', type=int, default=119223)

args = parser.parse_args()

lora_path = args.lora_path
save_dir = args.save_dir
seed = args.seed
prompt = args.prompt
os.makedirs(f"{save_dir}", exist_ok=True)


pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(lora_path) # change this.
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU





pipe.enable_vae_tiling()
    
image = pipe(prompt, 
            height=720,
            width=1440,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=50, 
            blend_extend=2,
            guidance_scale=7).images[0]

image = image.resize((2048,1024))

image.save(f"{save_dir}/rgb.png")
