import numpy as np
import torch
import cv2
import os
from pathlib import Path

torch.cuda.set_device(1)
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


from sam2.sam2.build_sam import build_sam2
from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import json

def mask_image(image, mask):
    """Masks an image with a binary mask, retaining color in the masked area and setting
       the rest to white.

    Args:
        image: The input image as a NumPy array.
        mask: The binary mask as a NumPy array, where 255 represents the masked area.

    Returns:
        The masked image as a NumPy array.
    """

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image[mask == 0] = 255  # Set unmasked areas to white
    return masked_image


def save_mask(anns, path):

    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for i, ann in enumerate(anns):
        #a = ann['original_index']
        mask = ann['segmentation']
        mask = np.stack([mask]*3, axis=-1)   #如果不进行remove处理，这句不用注释

        img = (mask*255).astype(np.uint8)  # Setting mask as white
        cv2.imwrite(f'{path}/mask_{i}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def show_anns(anns, image, save_path, borders=True):
    if len(anns) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        # 标注掩码的索引
        y, x = np.mean(np.argwhere(m), axis=0).astype(int)
        ax.text(x, y, str(i), color='white', fontsize=15, ha='center', va='center', weight='bold')

    ax.imshow(img)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close(fig)



def mask_iou(mask1, mask2):  
    # Compute IoU for two masks  
    intersection = np.logical_and(mask1, mask2).astype(np.float32).sum()  
    union = np.logical_or(mask1, mask2).astype(np.float32).sum()  
    return intersection / union if union > 0 else 0.0  
  

def filter_masks_by_overlap(masks, threshold):
    masks_np = [np.array(mask['segmentation'], dtype=np.bool) for mask in masks]
    areas = [np.sum(mask) for mask in masks_np]
    keep = torch.ones(len(masks_np), dtype=torch.bool)
    scores = [mask['stability_score'] for mask in masks]
    keep = torch.ones(len(masks_np), dtype=torch.bool)

    # 遍历每个掩码
    for i in range(len(masks_np)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks_np)):
            if not keep[j]:
                continue
            
            # 计算交集和 IoU
            intersection = np.logical_and(masks_np[i], masks_np[j]).astype(np.float32).sum()
            smaller_area = min(areas[i], areas[j])
            if intersection > threshold * smaller_area:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    # 过滤后的掩码
    filtered_masks = [mask for idx, mask in enumerate(masks) if keep[idx]]
    
    return filtered_masks


def crop_object_from_white_background(image):
   """Crops an image with a white background to the minimal bounding box containing a non-white object.
   """

   img = Image.fromarray(image)

   # Load the image
   img_array = np.array(image)

   # Find non-white pixels
   non_white_mask = np.any(img_array != 255, axis=2)  # Check all color channels

   # Find bounding box coordinates
   ymin, xmin = np.where(non_white_mask)[0].min(), np.where(non_white_mask)[1].min()
   ymax, xmax = np.where(non_white_mask)[0].max() + 1, np.where(non_white_mask)[1].max() + 1

   # Crop the image
   cropped_img = img.crop((xmin, ymin, xmax, ymax))

   return cropped_img, xmin, ymin, xmax, ymax


def convert_to_serializable(ann):
    """Convert annotation to a JSON-serializable format."""
    if isinstance(ann, dict):
        return {k: convert_to_serializable(v) for k, v in ann.items()}
    elif isinstance(ann, list):
        return [convert_to_serializable(i) for i in ann]
    elif isinstance(ann, np.ndarray):
        return ann.tolist()
    elif isinstance(ann, np.generic):
        return ann.item()
    else:
        return ann

def save_annotations(anns, path):
    for i, ann in enumerate(anns):
        simplified_ann = {
            "area": ann['area'],
            "bbox": ann['bbox'],
            "predicted_iou": ann['predicted_iou'],
            "point_coords": ann['point_coords'],
            "stability_score": ann['stability_score'],
            "crop_box": ann['crop_box']
        }
        ann_serializable = convert_to_serializable(simplified_ann)
        with open(f'{path}/mask_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(ann_serializable, f, ensure_ascii=False, indent=2)


def get_masked_image(mask_img_path):
    
    mask_img = cv2.imread(mask_img_path)[:, :, 0] # only one layer mask is needed
    #print("mask_img_path: ", mask_img_path)
    masked_image = mask_image(rgb_image, mask_img)
    return masked_image


def clip_prediction(image_input, texts, labels):
    text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    label = labels[np.argmax(similarity)]
    return label




#草莓的提示词  ##prompt of strawberry
texts = [
"a red strawberry",  #a red strawberry with numerous points
"a pale green strawberry with numerous points",
"a green veined strawberry leaf",
"a long and thin stem",
"a white flower",
"soil or background or something else",
]
labels = ['ripe', 'unripe', 'leaf','stem','flower','others']
label_dict = {"ripe": 0, "unripe": 1, "leaf": 2, "stem": 3, "flower": 4,"others": 5}



sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

image_segs_folder = "./Images"
masks_segs_folder = './output/strawberry/mask'
output_path = './output/strawberry/labels'
va_output_path = '../output/strawberry/visual_new'
va_all_output_path = './output/strawberry/visual_all'
json_save_dir = './output/strawberry/json'
mask_nms_key = True #默认是True，不需要改为False  ##Default is True, need to be changed to False
mask_nms_thresh = 0.9  #两个掩码重叠的面积占小掩码的阈值  ##The threshold of the area of two masks overlapping is the area of the smaller mask
print(f'Your mask_nms_key is {mask_nms_key} !')

sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"


sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_side=32, min_mask_region_area=50)
for img_tain_folder in os.listdir(image_segs_folder):
    #train val test
    img_files = os.listdir(os.path.join(image_segs_folder, img_tain_folder))
    for img_file in img_files:
        #if Path(img_file).stem in os.listdir('/home/nya/code/segment-anything-2/sam2_clip/out_peach/mask/train'):
            #continue
        #else:
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
            #show_anns(sorted_anns, image, path_stem_visual_all)
            #save_annotations(sorted_anns, json_save_path)
            save_mask(sorted_anns, path_stem)
            del image, masks2
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error with file {img_file}: {e}")
            continue

#生成大规模txt时
for img_train_folder in os.listdir(image_segs_folder):
    if img_train_folder == 'train':
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
            for file in os.listdir(mask_seg_folder):
                mask_path = os.path.join(mask_seg_folder, file)
                mask = cv2.imread(mask_path, 0)
                labelled_mask, num_labels = label_region(mask)
                region_sizes = np.bincount(labelled_mask.flat)
                region_sizes[0] = 0

                mask_img = cv2.imread(mask_path)[:, :, 0]
                masked_image = mask_image(rgb_image, mask_img)
                

                try:
                    masked_image = get_masked_image(mask_path)
                    image, xmin, ymin, xmax, ymax = crop_object_from_white_background(masked_image)
                    
                    image_preprocessed = preprocess(image)
                    image_input = torch.tensor(np.stack([image_preprocessed]))
                    label = clip_prediction(image_input, texts, labels)
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
                        skip = num_points // 300
                        skip = max(1, skip)
                        approx_sparse = c[::skip]
                        bottom_point_index = np.argmax(approx_sparse[:, 1])
                        sorted_points = np.concatenate([approx_sparse[bottom_point_index:], approx_sparse[:bottom_point_index]])
                        line += ' ' + ' '.join(f'{format(point[0]/img_width, ".6f")} {format(point[1]/img_height, ".6f")}' for point in sorted_points)
                        #print(len(sorted_points))
                        #print(sorted_points)

                    #file_contents[-1] = file_contents[-1].strip() + '\n'  # Remove extra space after the last label
                    line += '\n'
                    file_contents.append(line)

                except Exception as e:
                    print(f"Error processing file {mask_path}, skipping. Error was {e}")
                    continue

                filename = os.path.join(output_path, img_train_folder, f'{stem}.txt')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as f:
                    f.writelines(file_contents)

            if visual:
                img_final = cv2.imread(img_path)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                thickness = 5
                for res in results:
                    if res['label'] == 'ripe' or res['label'] == 'unripe':
                        cv2.rectangle(img_final, (res['xmin'], res['ymin']), (res['xmax'], res['ymax']), (76, 94, 229), 7)  # Red rectangles

                        # Add label with white background
                        (label_width, label_height), baseline = cv2.getTextSize(res['label'], font, font_scale, thickness)
                        top_left = (res['xmin'], res['ymin'] - label_height - baseline)
                        bottom_right = (res['xmin'] + label_width, res['ymin'] - baseline)
                        cv2.rectangle(img_final, top_left, bottom_right, (255, 255, 255), cv2.FILLED)
                
                        # Add the text label with precise alignment
                        text_origin = (res['xmin'], res['ymin'] - baseline)
                        cv2.putText(img_final, res['label'], text_origin, font, font_scale, (76, 94, 229), thickness)                        
                visual_dir = os.path.join(va_output_path, img_train_folder)
                os.makedirs(visual_dir, exist_ok=True)
                
                cv2.imwrite(os.path.join(visual_dir, img_file), img_final)

            print(filename,'  have been finished!')