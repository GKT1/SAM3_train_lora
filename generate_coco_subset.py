import os
import cv2
import json
import numpy as np
import random
import shutil
from pathlib import Path
from pycocotools import mask as mask_utils

random.seed(42)

# Define categories
categories_map = {
    'damaged building': [
        'train_building_damaged_mask',
        'train_building_destroyed_mask'
    ],
    'intact building': [
        'train_building_intact_mask'
    ],
    'road damage': [
        'train_road_debris_covered_mask',
        'train_road_flooded_mask'
    ],
    'intact road': [
        'train_road_intact_mask'
    ]
}

category_id_map = {
    'damaged building': 1,
    'intact building': 2,
    'road damage': 3,
    'intact road': 4
}

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate COCO subset from DisasterM3 dataset")
    parser.add_argument("--data_dir", type=str, default="../DisasterM3", help="Path to original DisasterM3 dataset")
    parser.add_argument("--out_dir", type=str, default="disaster_subset_coco", help="Output directory for COCO dataset")
    parser.add_argument("--num_images", type=int, default=200, help="Target number of images to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--include_sar", action="store_true", help="Include SAR (radar) images (default: false, only optical used)")
    parser.add_argument("--filter_json", type=str, default="", help="Path to a filtered JSON file to select specific images")
    return parser.parse_args()

def setup_dirs(out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train').mkdir(parents=True, exist_ok=True)
    (out_dir / 'valid').mkdir(parents=True, exist_ok=True)

def find_masks(data_dir, include_sar=False):
    all_masks = []
    # Key is image base name, value is list of masks and category
    image_to_masks = {}
    
    for category, mask_dirs in categories_map.items():
        for mask_dir in mask_dirs:
            dir_path = data_dir / 'masks' / mask_dir
            if dir_path.exists():
                for mask_file in dir_path.glob('*.png'):
                    img_name = mask_file.stem
                    
                    if not include_sar and "sar" in img_name.lower():
                        continue
                        
                    if img_name not in image_to_masks:
                        image_to_masks[img_name] = []
                    image_to_masks[img_name].append({
                        'category': category,
                        'mask_path': mask_file
                    })
    
    return image_to_masks

def create_coco_structure():
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "damaged building"},
            {"id": 2, "name": "intact building"},
            {"id": 3, "name": "road damage"},
            {"id": 4, "name": "intact road"}
        ]
    }

def process(data_dir, out_dir, target_images, include_sar=False, filter_json_path=""):
    setup_dirs(out_dir)
    image_to_masks = find_masks(data_dir, include_sar)
    
    # Get all valid unique image base names
    all_images = list(image_to_masks.keys())
    
    selected_images = all_images
    
    # Filter by JSON if provided
    if filter_json_path and os.path.exists(filter_json_path):
        print(f"Filtering images based on {filter_json_path}")
        with open(filter_json_path, 'r') as f:
            filtered_data = json.load(f)
        
        # Extract base names from the JSON. 
        # e.g. "train_images\\bata_explosion_post_0.png" -> "bata_explosion_post_0"
        allowed_bases = set()
        for item in filtered_data:
            post_path = item.get("post_image_path", "")
            if post_path:
                # Handle Windows and Unix slashes
                filename = post_path.replace('\\', '/').split('/')[-1]
                base = filename.rsplit('.', 1)[0]
                base = base.replace('_post_disaster', '').replace('_post', '')
                allowed_bases.add(base)
                
                pre_path = item.get("pre_image_path", "")
                if pre_path:
                    pre_filename = pre_path.replace('\\', '/').split('/')[-1]
                    pre_base = pre_filename.rsplit('.', 1)[0]
                    pre_base = pre_base.replace('_pre_disaster', '').replace('_pre', '')
                    allowed_bases.add(pre_base)
                    
        # Filter all_images
        selected_images = [img for img in all_images if img in allowed_bases]
        print(f"Images after JSON filtering: {len(selected_images)} (from {len(all_images)} found in masks)")
        target_images = len(selected_images)
    else:
        # Shuffle and pick target_images
        random.shuffle(all_images)
        
        # If target is larger than available, use all
        if target_images > len(all_images):
            print(f"Requested {target_images} images but only {len(all_images)} available. Using all available images.")
            target_images = len(all_images)
            
        selected_images = all_images[:target_images]
    
    # Split: 80% train, 20% valid
    split_idx = int(len(selected_images) * 0.8)
    train_images = selected_images[:split_idx]
    valid_images = selected_images[split_idx:]
    
    print(f"Total selected: {len(selected_images)}")
    print(f"Train: {len(train_images)}, Valid: {len(valid_images)}")
    
    # Helper to process a split
    def process_split(split_name, img_list):
        coco_data = create_coco_structure()
        split_dir = out_dir / split_name
        
        annotation_id = 1
        image_id = 1
        
        for img_name in img_list:
            # Find the actual image file
            img_candidates = [
                data_dir / 'train_images' / f"{img_name}.png",
                data_dir / 'train_images' / f"{img_name}_post_disaster.png",
                data_dir / 'train_images' / f"{img_name.replace('_post_disaster', '')}.png",
                data_dir / 'train_images' / f"{img_name.replace('_post_disaster', '')}_post_disaster.png",
            ]
            
            img_path = None
            for cand in img_candidates:
                if cand.exists():
                    img_path = cand
                    break
                    
            if img_path is None:
                continue
                
            # Copy image to split dir
            out_img_path = split_dir / img_path.name
            if not out_img_path.exists():
                shutil.copy2(img_path, out_img_path)
            
            # Read image to get dimensions (using cv2 just for shape)
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            height, width = img_cv.shape[:2]
            
            # Add to COCO images list
            coco_data['images'].append({
                "id": image_id,
                "file_name": out_img_path.name,
                "height": height,
                "width": width
            })
            
            # Process all masks for this image
            masks_info = image_to_masks[img_name]
            for m_info in masks_info:
                mask_path = m_info['mask_path']
                category = m_info['category']
                cat_id = category_id_map[category]
                
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                    
                binary_mask = (mask > 0).astype(np.uint8)
                y_indices, x_indices = np.where(binary_mask > 0)
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue
                    
                x1, x2 = int(np.min(x_indices)), int(np.max(x_indices))
                y1, y2 = int(np.min(y_indices)), int(np.max(y_indices))
                w = x2 - x1
                h = y2 - y1
                
                # Encode mask using pycocotools
                encoded_mask = mask_utils.encode(np.asfortranarray(binary_mask))
                encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
                area = float(mask_utils.area(encoded_mask))
                
                coco_data['annotations'].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, w, h],
                    "area": area,
                    "segmentation": encoded_mask,
                    "iscrowd": 0
                })
                annotation_id += 1
                
            image_id += 1
            
        # Write annotation file
        ann_file = split_dir / '_annotations.coco.json'
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f)
            
        print(f"Processed {split_name}: {image_id-1} images, {annotation_id-1} annotations.")

    process_split('train', train_images)
    process_split('valid', valid_images)

if __name__ == '__main__':
    args = parse_args()
    random.seed(args.seed)
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    
    process(data_dir, out_dir, args.num_images, args.include_sar, args.filter_json)
    print("COCO dataset generation complete!")
