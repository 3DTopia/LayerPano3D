## [Rendering Visualization]

save_dir="outputs"
elevation=0


python -m rendering.render_video_360 \
            --save_dir "${save_dir}/scene" \
            --elevation ${elevation}
            
python -m rendering.render_video_zigzag \
        --save_dir "${save_dir}/scene"
