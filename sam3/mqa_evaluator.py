import os
import json
import torch
import numpy as np
import pycocotools.mask as mask_utils
import sys
import re
from PIL import Image
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.train.masks_ops import rle_encode

def parse_option(opt_str):
    s = str(opt_str).replace('%', '')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match:
        return float(match.group())
    return None

def closest_option(computed_val, options_list):
    closest_idx = 0
    min_diff = float('inf')
    for i, opt in enumerate(options_list):
        val = parse_option(opt)
        if val is not None:
            diff = abs(val - computed_val)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
    return chr(ord('A') + closest_idx), options_list[closest_idx]

def get_union_area_percentage(pred_masks, orig_h, orig_w):
    if len(pred_masks) == 0: return 0.0
    rle_masks = [{"size": (orig_h, orig_w), "counts": rle} for rle in pred_masks]
    binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
    union_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for m in binary_masks:
        union_mask = np.logical_or(union_mask, m).astype(np.uint8)
    total_area = np.sum(union_mask)
    return (total_area / (orig_h * orig_w)) * 100.0

def evaluate_mqa_on_dataset(model, device, scenarios_path, images_dir, threshold=0.25):
    """
    Evaluates Multiple Choice Accuracy (MQA) and Mean Absolute Error on a dataset.
    Automatically infers the prompt from the scenario description.
    """
    if not os.path.exists(scenarios_path):
        raise FileNotFoundError(f"Scenarios file not found: {scenarios_path}")
        
    with open(scenarios_path, "r") as f:
        scenarios = json.load(f)
        
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
    
    diffs = []
    corrects = 0
    
    for scenario in tqdm(scenarios, desc=f"MQA Eval ({os.path.basename(scenarios_path)})"):
        post_img_name = os.path.basename(scenario["post_image_path"].replace("\\", "/"))
        
        # Skip SAR images to match optical-only training
        if "sar" in post_img_name.lower():
            continue
            
        post_img = os.path.join(images_dir, post_img_name)
        
        if not os.path.exists(post_img):
            continue
            
        # Robust ground truth parsing (handle int, float, or string like "25%")
        gt_raw = scenario["ground_truth"]
        if isinstance(gt_raw, (int, float)):
            gt_val = float(gt_raw)
        else:
            gt_val = float(str(gt_raw).replace('%', ''))
            
        gt_option = scenario["ground_truth_option"]
        
        # Determine prompt dynamically based on task
        cls_desc = scenario.get("cls_description", "").lower()
        if "intact road" in cls_desc:
            dynamic_prompt = "intact road"
        elif "road" in cls_desc:
            dynamic_prompt = "road damage"
        elif "building" in cls_desc:
            # Use simple prompt "building" for all building counting tasks
            dynamic_prompt = "building"
        else:
            dynamic_prompt = "object"
            
        # Inference
        image = Image.open(post_img)
        orig_w, orig_h = image.size
        
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=dynamic_prompt)
        
        if state["masks"].shape[0] > 0:
            # We skip NMS entirely to avoid OOM and rely on area-based heuristics
            # for counting dense objects. SAM3 object queries are already
            # designed to be largely distinct.
            pred_masks = rle_encode(state["masks"].squeeze(1))
            pred_masks = [m["counts"] for m in pred_masks]
        else:
            pred_masks = []
            
        is_percentage = "%" in scenario.get("options_str", "")
        
        if is_percentage:
            pred_val = get_union_area_percentage(pred_masks, orig_h, orig_w)
        elif "building" in cls_desc:
            # We trust the model's distinct object queries to count houses.
            # Removing the area-based heuristic as it was too brittle.
            pred_val = len(pred_masks)
        else:
            pred_val = len(pred_masks)
            
        diff = abs(pred_val - gt_val)
        diffs.append(diff)
        
        pred_option, _ = closest_option(pred_val, scenario["options_list"])
        if pred_option == gt_option:
            corrects += 1
            
    num_eval = len(diffs)
    if num_eval == 0:
        return {"accuracy": 0.0, "mae": 0.0}
        
    return {
        "accuracy": corrects / num_eval,
        "mae": sum(diffs) / num_eval
    }
