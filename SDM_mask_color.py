#这份代码是您在使用SDM的同时灵活地调整掩码的颜色。
###This code is for you to flexibly adjust the color of the mask while using SDM.

import numpy as np
import torch
import cv2
import os
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath('./segment-anything-2'))  ##Add the path to the segment-anything-2 folder

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

def final_visualization(image, anns, results, save_path):
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)  
    ax.set_autoscale_on(False)

    # 创建一个 RGBA 图像用于掩码着色  ##Create an RGBA image for mask coloring
    img_with_masks = image.copy()  # 复制原始图像，用于显示掩码  ##Copy the original image for displaying the mask
    overlay = np.zeros_like(img_with_masks, dtype=np.uint8)

    # 为每个掩码区域着色，并保持颜色与标签一致  ##Color each mask area and keep the color consistent with the label
    for i, ann in enumerate(anns):
        mask = ann['segmentation']
        
        # 获取标签并设置颜色  ##Get the label and set the color
        label = ann['label']
        if label == 'others':
            color =  (252, 248, 187) # 注意这里是0-255范围的值 ##Note that this is a value in the 0-255 range
        elif label == 'waxberry':  
            color = (229,76,94)  # 粉色  ##Pink
        elif label == 'unripe':
            color = (146, 208, 80)  # 浅绿色  ##Light green
        elif label == 'leaf':
            color = (0, 176, 80)  # 绿色  ##Green
        elif label == 'stem':
            color = (243, 163, 97)  # 橙色  ##Orange
        elif label == 'flower':
            color = (168, 218, 219)  # 浅黄色   ##Light yellow

        # # 将掩码区域的颜色设置为对应的标签颜色  ##Set the color of the target area to the corresponding label color
        overlay[mask > 0] = color  

    alpha = 0.4  # 掩码透明度  ##Mask transparency
    img_with_masks = cv2.addWeighted(overlay, alpha, img_with_masks, 1 - alpha, 0)

    # 显示带有掩码的图像  ##Display the image with the mask
    ax.imshow(img_with_masks)

    # 在图像上绘制边界框和标签  ##Draw the bounding box and label on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 2
    font_color = (1, 1, 1)  # 白色  ##White

    for res in results:
        # 设置每个标签的颜色 ##Set the color of each label
        if res['label'] == 'others':
            continue
            box_color =   (252/255, 248/255, 187/255)
            label_bg_color = (252/255, 248/255, 187/255)
        elif res['label'] == 'waxberry':
            box_color =(229/255, 76/255, 94/255)  #粉色  ##Pink
            
            label_bg_color = (229/255, 76/255, 94/255)
        elif res['label'] == 'unripe':
            box_color = (146/255, 208/255, 80/255)  # 浅绿色  ##Light green
            label_bg_color = (146/255, 208/255, 80/255)
        elif res['label'] == 'leaf':
            box_color = (0/255, 176/255, 80/255)  # 绿色  ##Green
            label_bg_color = (0/255, 176/255, 80/255)
        elif res['label'] == 'stem':
            box_color = (243/255, 163/255, 97/255)  # 橙色  ##Orange
            label_bg_color = (243/255, 163/255, 97/255)
        elif res['label'] == 'flower':
            box_color = (168/255, 218/255, 219/255)  # 浅黄色  ##Light yellow
            label_bg_color = (168/255, 218/255, 219/255)

        # 绘制矩形框  ##Draw the rectangle
        rect = plt.Rectangle((res['xmin'], res['ymin']),
                             res['xmax'] - res['xmin'],
                             res['ymax'] - res['ymin'],
                             linewidth=2, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect)

        # 绘制填充文本框  ##Draw the filled text box
        ax.text(res['xmin'], res['ymin'] - 5, res['label'], color='white', fontsize=30,
                ha='left', va='bottom', bbox=dict(facecolor=label_bg_color, edgecolor='none', boxstyle='round,pad=0'))

    # 保存最终的图像  ##Save the final image
    plt.axis('off')
    plt.savefig(save_path)
    print(save_path)
    plt.close(fig)



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
        mask = np.stack([mask]*3, axis=-1)  

        img = (mask*255).astype(np.uint8)  
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

    # 遍历每个掩码  ##Iterate over each mask
    for i in range(len(masks_np)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(masks_np)):
            if not keep[j]:
                continue
            
            # 计算交集和 IoU  ##Calculate the intersection and IoU
            intersection = np.logical_and(masks_np[i], masks_np[j]).astype(np.float32).sum()
            smaller_area = min(areas[i], areas[j])
            if intersection > threshold * smaller_area:
                if scores[i] < scores[j]:
                    keep[i] = False
                else:
                    keep[j] = False

    # 过滤后的掩码  ##Filtered masks
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
    
    mask_img = cv2.imread(mask_img_path)[:, :, 0] 
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
    #print('第1轮标签', label)
    return label



#草莓的提示词  ##prompt of strawberry
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
mask_nms_key = True #默认是True，不需要改为False  ##Default is True, need to be changed to False
mask_nms_thresh = 0.9  #两个掩码重叠的面积占小掩码的阈值  ##The threshold of the area of two masks overlapping is the area of the smaller mask
print(f'Your mask_nms_key is {mask_nms_key} !')

sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
#根据你的数据集调整min_mask_region_area  ##Adjust min_mask_region_area according to the your dataset
mask_generator = SAM2AutomaticMaskGenerator(sam2, points_per_side=32, min_mask_region_area=50) 

#SAM2生成掩码  ##SAM2 generates masks
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



#OpenClip对齐  ##OpenClip alignment
for img_train_folder in os.listdir(image_segs_folder):
    if img_train_folder == 'waxberry':

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
                        skip = num_points // 300  #根据需求灵活调整取点数  ##Adjust the number of points according to the demand
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

            # 保存最终的可视化结果  ##Save the final visualization result
            visual_dir = os.path.join(va_output_path, img_train_folder)
            os.makedirs(visual_dir, exist_ok=True)
            final_visualization(rgb_image, masks, results, os.path.join(visual_dir, img_file))

            print(filename, '  have been finished!')
    else:
        continue