import os
import tqdm
import argparse
from PIL import Image
from loguru import logger
from datetime import datetime
import shutil
from LayerPano import LayerPano
# from PanoScene4D import PanoScene4D

def init_logger():
    logger.remove()  # Remove default logger
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>"
    logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True, format=log_format)



if __name__ == "__main__":
    ### option
    parser = argparse.ArgumentParser(description='Arguments for LayerPano')
    # Save options
    parser.add_argument('--save_dir',  type=str, default='outputs', help='Save directory')
    parser.add_argument('--input_dir', type=str, default='outputs')
    parser.add_argument('--outlier_thresh', type=int, default=4, help='get outlier mask threshold')
    args = parser.parse_args()
    
    save_dir = f"{args.save_dir}/scene"

    init_logger()
    os.makedirs(f"{save_dir}", exist_ok=True)
    
    logger.info(f"[INFO] SAVE_PATH: {save_dir}")
    logger.info(f"[INFO] INPUT_PATH: {args.input_dir}")
    logger.info("[INFO] START TRAINING")
    

    layerpano = LayerPano(save_dir=save_dir)
    layerpano.create(args.input_dir, outlier_thresh=args.outlier_thresh)
