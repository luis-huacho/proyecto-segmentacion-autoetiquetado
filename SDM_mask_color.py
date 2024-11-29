###This code is for you to flexibly adjust the color of the mask while using SDM.
import argparse
import numpy as np
import torch
import cv2
import os
from pathlib import Path

import sys
import sys
sys.path.append('your/path/to/SDM-D/sam2')  ##Add the path to the sam2 folder

torch.cuda.set_device(0)
#print(torch.cuda.current_device())

import random
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from scipy.ndimage import label as label_region
from open_clip import tokenizer
import open_clip
open_clip.list_pretrained()
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json

from utils import read_strawberry_descriptions, create_output_folders
from utils import generate_all_sam_mask, label_assignment







  











##prompt of strawberry
texts = [
"a red strawberry with numerous points",
"a pale green strawberry with numerous points",
"a green veined leaf with white points",
"a long and thin stem",
"a white flower",
"soil or background or something else",
]
labels = ['ripe', 'unripe', 'leaf','stem','flower','others']
label_dict = {"ripe": 0, "unripe": 1, "leaf": 2, "stem": 3, "flower": 4,"others": 5}



sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

image_segs_folder = "../Images"
masks_segs_folder = '../output/strawberry/mask'
output_path = '../output/strawberry/labels'
va_output_path = '../output/strawberry/visual_new'
va_all_output_path = '../output/strawberry/visual_all'
json_save_dir = '../output/strawberry/json'
mask_nms_key = True ##Default is True, need to be changed to False
mask_nms_thresh = 0.9  ##The threshold of the area of two masks overlapping is the area of the smaller mask
print(f'Your mask_nms_key is {mask_nms_key} !')

sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
##Adjust min_mask_region_area according to the your dataset
mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_side=32, min_mask_region_area=50) 

##SAM2 generates masks
for img_tain_folder in os.listdir(image_segs_folder):
    img_files = os.listdir(os.path.join(image_segs_folder, img_tain_folder))
    for img_file in img_files:
        img_path = os.path.join(image_segs_folder, img_tain_folder, img_file)
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            stem, suffix = os.path.splitext(img_file)
            os.makedirs(f'{masks_segs_folder}/{img_tain_folder}/{stem}', exist_ok=True)
            path_stem = f'{masks_segs_folder}/{img_tain_folder}/{stem}'
            os.makedirs(f'{va_all_output_path}/{img_tain_folder}', exist_ok=True)
            path_stem_visual_all = f'{va_all_output_path}/{img_tain_folder}/{stem}'
            os.makedirs(f'{json_save_dir}/{img_tain_folder}/{stem}', exist_ok=True)
            json_save_path = f'{json_save_dir}/{img_tain_folder}/{stem}'
                            
            masks2 = mask_generator.generate(image)
            sorted_anns = sorted(masks2, key=(lambda x: x['area']), reverse=True)
            if mask_nms_key:
                sorted_anns = filter_masks_by_overlap(sorted_anns, mask_nms_thresh)
            show_anns(sorted_anns, image, path_stem_visual_all)  #保存每张图的分割结果，不需要可以注释  ##Save the segmentation result of each image, if not needed, can be commented
            save_annotations(sorted_anns, json_save_path)  #保存每个掩码的json,不需要可以注释  ##Save the json of each mask, if not needed, can be commented
            save_mask(sorted_anns, path_stem)
            del image, masks2
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with file {img_file}: {e}")
            continue



##OpenClip alignment
for img_train_folder in os.listdir(image_segs_folder):

    img_files = os.listdir(os.path.join(image_segs_folder, img_train_folder))
    for img_file in img_files:
        img_file_with_txt_suffix = Path(img_file).with_suffix('.txt')
        stem, suffix = os.path.splitext(img_file)
        img_path = os.path.join(image_segs_folder, img_train_folder, img_file)
        image = Image.open(img_path).convert('RGB')
        rgb_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_width, img_height = image.size
        results = []
        mask_seg_folder = os.path.join(masks_segs_folder, img_train_folder, stem)
        file_contents = []
        masks = []

        for file in os.listdir(mask_seg_folder):
            mask_path = os.path.join(mask_seg_folder, file)
            mask = cv2.imread(mask_path, 0)
            labelled_mask, num_labels = label_region(mask)
            region_sizes = np.bincount(labelled_mask.flat)
            region_sizes[0] = 0
            mask_img = cv2.imread(mask_path)[:, :, 0]
            masked_image = mask_image(rgb_image, mask_img)
            try:
                masked_image = get_masked_image(rgb_image, mask_path)
                image, xmin, ymin, xmax, ymax = crop_object_from_white_background(masked_image)
                
                image_preprocessed = preprocess(image)
                image_input = torch.tensor(np.stack([image_preprocessed]))
                label = clip_prediction(model, image_input, texts, labels)
                label_num = label_dict[label]
                results.append({"label": label, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
                #file_contents.append(f'{label_num} ')
                line = f'{label_num}'

                for region_label in range(1, num_labels+1):
                    mask_cur = ((labelled_mask == region_label) * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_cur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    c = max(contours, key=cv2.contourArea)
                    c = c.reshape(-1, 2)
                    num_points = len(c)
                    skip = num_points // 300  ##Adjust the number of points according to the demand
                    skip = max(1, skip)
                    approx_sparse = c[::skip]
                    bottom_point_index = np.argmax(approx_sparse[:, 1])
                    sorted_points = np.concatenate([approx_sparse[bottom_point_index:], approx_sparse[:bottom_point_index]])
                    line += ' ' + ' '.join(f'{format(point[0]/img_width, ".6f")} {format(point[1]/img_height, ".6f")}' for point in sorted_points)
                line += '\n'
                file_contents.append(line)

                masks.append({
                    'segmentation': mask_img,
                    'area': np.sum(mask_img),
                    'label': label
                })

            except Exception as e:
                print(f"Error processing file {mask_path}, skipping. Error was {e}")
                continue

        filename = os.path.join(output_path, img_train_folder, f'{stem}.txt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.writelines(file_contents)

        ##Save the final visualization result
        visual_dir = os.path.join(va_output_path, img_train_folder)
        os.makedirs(visual_dir, exist_ok=True)
        mask_color_visualization(rgb_image, masks, results, os.path.join(visual_dir, img_file))

        print(filename, '  have been finished!')