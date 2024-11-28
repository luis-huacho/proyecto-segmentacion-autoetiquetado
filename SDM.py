import argparse
import numpy as np
import torch
import cv2
import os
from pathlib import Path
import sys
sys.path.append('your/path/to/SDM-D/sam2')

#print(torch.cuda.current_device())

import random
from PIL import Image
from collections import OrderedDict
import open_clip

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from utils import mask_image, save_mask, show_anns, filter_masks_by_overlap, crop_object_from_white_background, save_annotations, get_masked_image, clip_prediction, read_strawberry_descriptions, create_output_folders
from utils import generate_all_sam_mask, label_assignment

# Parse arguments
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='./Images/strawberry', required = True, help='Path to the image segmentation folder')
    parser.add_argument('--out_folder', type=str, default='./output/strawberry', required = True,help='Path to save mask outputs')
    parser.add_argument('--des_file', type=str, default='./description/straw_des.txt', required = True,help='Path to your prompt texts')
    parser.add_argument('--sam2_checkpoint', type=str, default="./checkpoints/sam2_hiera_large.pt", required = False, help='SAM2 model checkpoint path')
    parser.add_argument('--model_cfg', type=str, default="sam2_hiera_l.yaml", required = False, help='SAM2 model config file')
    parser.add_argument('--enable_mask_nms', type=bool, default=True, required = False,  help='Whether to apply NMS to masks')
    parser.add_argument('--mask_nms_thresh', type=float, default=0.9, required = False, help='Threshold for NMS mask overlap')
    parser.add_argument('--save_anns', type=bool, default=True, required = False,  help='Whether to save mask anns')
    parser.add_argument('--save_json', type=bool, default=True, required = False,  help='Whether to save json')
    parser.add_argument('--visual', type=bool, default=True, required = False,  help='Whether to visual results')
    return parser.parse_args()


def main():
    opt = parse_opt()
    # generate folder dirs
    image_folder = opt.image_folder
    out_folder = opt.out_folder
    enable_mask_nms = opt.enable_mask_nms
    save_anns = opt.save_anns
    save_json = opt.save_json
    mask_nms_thresh = opt.mask_nms_thresh
    masks_segs_folder = os.path.join(out_folder, 'mask')
    json_save_dir = os.path.join(out_folder, 'json')
    output_path = os.path.join(out_folder, 'labels')
    vis_output_path = os.path.join(out_folder, 'visual')
    label_out_dir = os.path.join(out_folder, 'label_visual')
    create_output_folders(out_folder)
    texts, labels, label_dict = read_strawberry_descriptions(opt.des_file)  

    # Init openCLIP model
    torch.cuda.set_device(0)
    clip_model, _, clip_preprocessor = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Init SAM2 model    
    sam2 = build_sam2(opt.model_cfg, opt.sam2_checkpoint, device='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_side=32, min_mask_region_area=50)

    print(f'Your enable_mask_nms is {opt.enable_mask_nms} !')

    # generate all masks
    generate_all_sam_mask(mask_generator, image_folder, masks_segs_folder, json_save_dir, vis_output_path, enable_mask_nms, mask_nms_thresh, save_anns, save_json)

    # label assignment
    label_assignment(clip_preprocessor, image_folder, masks_segs_folder, output_path, vis_output_path, label_out_dir, clip_model, texts, labels, label_dict, opt)

if __name__ == '__main__':
    main()