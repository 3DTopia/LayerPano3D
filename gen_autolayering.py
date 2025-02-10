# from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
import torch
import os
from PIL import Image
import requests
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from src.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2

def load_depth_model(ckpt_path):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.cuda().eval()
    return model



def run_depth(model, image):
    image = image.astype(np.uint8)
    erp_pred = model.infer_image(image)    
    output = (erp_pred -  erp_pred.min()) / (erp_pred.max() - erp_pred.min())
    output = (output* 65535).astype(np.int32)
    return output



def draw_mask_overlay(mask, image):
    rand_color = np.array([random.randint(0, 255) for _ in range(3)])
    colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    colored_mask = colored_mask * rand_color
    colored_mask = colored_mask.astype(np.uint8)

    # Blend the image and the colored mask
    alpha = 0.6  # Transparency factor
    overlay = cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)
    return overlay

def get_smooth_mask(general_mask, ksize = 16):

    mask_array = np.zeros_like(general_mask)
    height, width = general_mask.shape
    for i in range(height-2*ksize-1):
        for j in range(width-2*ksize-1):
            flag = general_mask[i+ksize,j+ksize]
            if flag != general_mask[i+ksize,j+ksize]:          
                print(flag, general_mask[i+ksize,j+ksize], i+ksize,j+ksize)
            if flag:
                mask_array[i:i+2*ksize+1,j:j+2*ksize+1] = True

    return mask_array

# the Auto API loads a OneFormerProcessor for us, based on the checkpoint
processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
depth_model = load_depth_model(ckpt_path='checkpoints/depth_anything_v2_vitl.pth')
from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description="Step 0 Parameters")

parser.add_argument("--input_dir", default="", type=str)
parser.add_argument("--scene_type", default="indoor", type=str) #"indoor" or "outdoor"
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

scene_type = args.scene_type
input_dir = f"{args.input_dir}/layering"

img_path = os.path.join(input_dir, 'rgb.png')
panoptic_dir = input_dir

os.makedirs(panoptic_dir, exist_ok=True)

img_type = img_path.split('.')[-1]
image = Image.open(img_path).convert('RGB')
image_array = np.array(image)

depth_array = run_depth(depth_model, image_array)

depth_array_pil = Image.fromarray(depth_array)
depth_array_pil.save(os.path.join(input_dir, 'depth_pred.png'))


w,h = image.size

if scene_type == "indoor": # indoor
    bg_labels = [87, 96, 97, 98, 99, 100, 102, 103, 105, 106, 113, 122, 123, 125, 126, 132, 118, 119, 131, 115, 116, 121,110,111,112, 86, 85, 93, 104] #indoor
elif scene_type == "outdoor": # outdoor
    bg_labels = [87, 96, 97, 98, 99, 100, 102, 103, 105, 106, 113, 122, 123, 125, 126, 132, 119, 88] # outdoor
# 
# {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'p
# arking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'um│·······························································································
# brella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skate│·······························································································
# board', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwic│·······························································································
# h', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', │·······························································································
# 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73:│·······························································································
#  'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush', 80: 'banner', 81: 'blanket', 82: 'bridge', 83: 'cardboard', 84: 'counter',│·······························································································
# #  85: 'curtain', 86: 'door-stuff', 87: 'floor-wood', 88: 'flower', 89: 'fruit', 90: 'gravel', 91: 'house', 92: 'light', 93: 'mirror-stuff', 94: 'net', 95: 'pillow', 96: 'platform'│·······························································································
# , 97: 'playingfield', 98: 'railroad', 99: 'river', 100: 'road', 101: 'roof', 102: 'sand', 103: 'sea', 104: 'shelf', 105: 'snow', 106: 'stairs', 107: 'tent', 108: 'towel', 109: 'w│·······························································································
# all-brick', 110: 'wall-stone', 111: 'wall-tile', 112: 'wall-wood', 113: 'water-other', 114: 'window-blind', 115: 'window-other', 116: 'tree-merged', 117: 'fence-merged', 118: 'ce│·······························································································
# iling-merged', 119: 'sky-other-merged', 120: 'cabinet-merged', 121: 'table-merged', 122: 'floor-other-merged', 123: 'pavement-merged', 124: 'mountain-merged', 125: 'grass-merged'│·······························································································
# , 126: 'dirt-merged', 127: 'paper-merged', 128: 'food-other-merged', 129: 'building-other-merged', 130: 'rock-merged', 131: 'wall-other-merged', 132: 'rug-merged'}   
# prepare image for the model
panoptic_inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
processor.tokenizer.batch_decode(panoptic_inputs.task_inputs)
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

with torch.no_grad():
    outputs = model(**panoptic_inputs)

panoptic_segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
segmentation = panoptic_segmentation['segmentation']
segments_info = panoptic_segmentation['segments_info']

labels = model.config.id2label
# print(labels)

def draw_panoptic_segmentation(segmentation, segments_info):
    # get the used color map
    viridis = cm.get_cmap('plasma', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    # for each segment, draw its legend
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        
    # ax.legend(handles=handles)
    plt.savefig(f'{panoptic_dir}/general_panoptic.png')

if args.debug:
    draw_panoptic_segmentation(**panoptic_segmentation)

panoptic_depths, panoptic_id, X, masks = [], [], [], []
bg_mask = np.zeros((h,w)).astype(bool)
for segment in segments_info:
    segment_id = segment['id']
    segment_label_id = segment['label_id']
    score = segment['score']
    
    segment_label = model.config.id2label[segment_label_id]
    
    
    mask = segmentation == segment_id
    mask = mask.detach().cpu().numpy()

    if segment_label_id in bg_labels:
        bg_mask = bg_mask | mask
        continue

    depth_masked_array = depth_array[mask]
    sorted_depth_array = np.sort(depth_masked_array)
    index = int(len(sorted_depth_array) * 0.25)
    top_25_percent = sorted_depth_array[index:]
    depth_value = np.mean(top_25_percent)

    if args.debug:
        image_mask = image_array * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        image_mask = Image.fromarray(image_mask.astype(np.uint8))
        image_mask.save(os.path.join(panoptic_dir, f'panoptic_{segment_label}_{segment_id}_{score}_{depth_value}.png'))

    
    # if score < 0.95:
    #     continue

    panoptic_depths.append(depth_value)
    panoptic_id.append((segment_label, segment_id))
    masks.append(mask)

    
    X.append((depth_value))

if args.debug:
    bg_overlay = draw_mask_overlay(bg_mask, image_array)
    bg_overay_pil = Image.fromarray(bg_overlay)
    bg_overay_pil.save(os.path.join(panoptic_dir, f'background_mask_overlay.png'))

X = np.asarray(X).reshape((-1,1))
sorted_indices = np.argsort(panoptic_depths)
if len(sorted_indices) == 0:
    print('Zero Instance Detected!!!!!')

else:

    print('Sorted Panoptic:', [panoptic_id[i] for i in sorted_indices])
    print('Sorted Panoptic Depth Levels:', [panoptic_depths[i] for i in sorted_indices])



    best_n_clusters = -1
    best_cost = 1e4
    max_cluster = 4
    if max_cluster > len(X)+1:
        max_cluster = len(X)+1

    print(f'==================== Kmeans in {max_cluster-1} clusters')

        
    kmeans = KMeans(n_clusters=max_cluster-1, random_state=42)
    kmeans.fit(X)
    cluster_centers = kmeans.cluster_centers_
    y_kmeans = kmeans.predict(X)
    print('KMeans:',[y_kmeans[i] for i in sorted_indices])


    cluster_centers = cluster_centers.reshape((-1))
    sorted_cluster_indices = np.argsort(cluster_centers.reshape((-1)))
    # print(sorted_cluster_indices)
    # print([cluster_centers[i] for i in sorted_cluster_indices])
    overlays = []
    overlays.append(image_array)
    output_dir = os.path.join(panoptic_dir)
    os.makedirs(output_dir, exist_ok=True)  
    for n_layer in range(len(sorted_cluster_indices)):
        n_layer_reverse = len(sorted_cluster_indices) - n_layer - 1
        layer_dir = os.path.join(output_dir, f'layer{n_layer_reverse}')
        os.makedirs(layer_dir, exist_ok=True)
        cluster_id = sorted_cluster_indices[n_layer]
        mask_binary = np.zeros((h,w)).astype(bool)
        for i in range(len(y_kmeans)):
            if y_kmeans[i] == cluster_id:
                mask_binary = mask_binary | masks[i]
        
        # mask_binary = mask_binary & (~bg_mask)
        mask_smooth = get_smooth_mask(mask_binary)

        mask_pil = Image.fromarray(mask_binary.astype(np.uint8)*255)
        mask_pil.save(f'{layer_dir}/layer{n_layer_reverse}_mask.png')

        mask_smooth_pil = Image.fromarray(mask_smooth.astype(np.uint8)*255)
        mask_smooth_pil.save(f'{layer_dir}/layer{n_layer_reverse}_mask_smooth.png')

        overlay = draw_mask_overlay(mask_binary, image_array)
        overlays.append(overlay)

    overlays = np.vstack(overlays)

    overlay_pil = Image.fromarray(overlays)
    overlay_pil.save(f'{panoptic_dir}/layer_mask_visualization.png')
