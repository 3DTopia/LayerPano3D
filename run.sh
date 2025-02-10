

# generation args
save_dir="outputs"
prompt="Sandy beach, large driftwood in the foreground, calm sea beyond, realism style."
scene_type="outdoor"
lora_path='checkpoints/pano_lora_720*1440_v1.safetensors'
depth_model='DepthAnythingv2'
seed=16806

# [step-1]: generate reference panorama
python gen_refpano.py \
    --save_dir ${save_dir} \
    --prompt "${prompt}" \
    --lora_path ${lora_path} \
    --seed ${seed} \

# [step-2]: autolayering
python gen_panodepth.py \
    --save_dir ${save_dir} \
    --depth_model ${depth_model} \
    --input_path "${save_dir}/rgb.png" \

python gen_autolayering.py \
    --input_dir ${save_dir} \
    --scene_type ${scene_type} \

# [step-3]: construct layered RGB panoramas
python gen_layerdata.py \
    --lora_path ${lora_path} \
    --base_dir "${save_dir}/layering" \
    --seed ${seed} \

# [optional SR step]
python pasd/run_layers_pasd.py \
    --inputs_dir "${save_dir}/layering" \
    --upscale 2 

# [step-3]: construct layered 3D panoramas(rgb+depth)


python gen_traindata.py \
    --save_dir ${save_dir} \
    --depth_model ${depth_model} \
    --layerpano_dir "${save_dir}/layering"  \
    --sr # use sr panoramas(if you run the above SR step)



# [step-4]: Panoramic 3D Gaussian Scene construction
python run_layerpano.py  \
            --input_dir ${save_dir} \
            --save_dir ${save_dir}


## Rendering args
elevation=0

## [Rendering Visualization]
python -m rendering.render_video_360 \
            --save_dir "${save_dir}/scene" \
            --elevation ${elevation}
            
python -m rendering.render_video_zigzag \
        --save_dir "${save_dir}/scene"


# apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/yangshuai/apptainer/panodepth.sif bash run.sh

