import os
import json
import torch
import numpy as np
import pycocotools.mask as mask_utils
import yaml
import sys
import re
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xyxy_to_xywh
from sam3.train.masks_ops import rle_encode

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SAM3_LoRA"))
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

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

def efficient_sam3_inference_multi_threshold(processor, image_path, text_prompt, thresholds):
    """Run SAM 3 image inference once, then evaluate multiple thresholds."""
    image = Image.open(image_path)
    orig_img_w, orig_img_h = image.size

    # Backbone inference
    state = processor.set_image(image)
    
    # Store results for each threshold
    results = {}
    
    # Run once to initialize text prompt and grounding
    state = processor.set_text_prompt(state=state, prompt=text_prompt)
    
    for t in thresholds:
        # Re-run grounding with new threshold
        state = processor.set_confidence_threshold(t, state=state)
        
        # format outputs
        if state["masks"].shape[0] > 0:
            pred_masks = rle_encode(state["masks"].squeeze(1))
            pred_masks = [m["counts"] for m in pred_masks]
        else:
            pred_masks = []
            
        results[t] = {
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_masks": pred_masks,
        }
        
    return results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model...")
    model = build_sam3_image_model(device=device, eval_mode=True, load_from_HF=True, enable_inst_interactivity=False)
    
    lora_config_path = "../SAM3_LoRA/configs/light_lora_config.yaml"
    lora_weights_path = "../SAM3_LoRA/outputs/sam3_lora_light/best_lora_weights.pt"
    
    with open(lora_config_path, 'r') as f:
        lora_cfg_dict = yaml.safe_load(f)
    lora_cfg = lora_cfg_dict["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=0.0,
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg.get("apply_to_vision_encoder", False),
        apply_to_text_encoder=lora_cfg.get("apply_to_text_encoder", False),
        apply_to_geometry_encoder=lora_cfg.get("apply_to_geometry_encoder", False),
        apply_to_detr_encoder=lora_cfg.get("apply_to_detr_encoder", False),
        apply_to_detr_decoder=lora_cfg.get("apply_to_detr_decoder", False),
        apply_to_mask_decoder=lora_cfg.get("apply_to_mask_decoder", False),
    )
    model = apply_lora_to_model(model, lora_config)
    load_lora_weights(model, lora_weights_path)
    model.to(device)
    model.eval()
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    
    with open("100_images_test/common_samples/sample_RDC.json", "r") as f:
        scenarios = json.load(f)
        
    test_images_dir = "100_images_test/test_images"
    
    print(f"\n--- Testing Thresholds on all {len(scenarios)} Images ---")
    
    # Track metrics
    diffs = {t: [] for t in thresholds}
    corrects = {t: 0 for t in thresholds}
    
    processor = Sam3Processor(model, device=device)
    
    for scenario in tqdm(scenarios, desc="Evaluating"):
        post_img_name = os.path.basename(scenario["post_image_path"].replace("\\", "/"))
        post_img = os.path.join(test_images_dir, post_img_name)
        
        if not os.path.exists(post_img):
            print(f"Skipping missing image: {post_img}")
            continue
            
        gt_str = scenario["ground_truth"].replace('%', '')
        gt_val = float(gt_str)
        gt_option = scenario["ground_truth_option"]
        
        # Efficient multi-threshold inference
        results = efficient_sam3_inference_multi_threshold(processor, post_img, "road", thresholds)
        
        for t in thresholds:
            res = results[t]
            pred_val = get_union_area_percentage(res["pred_masks"], res["orig_img_h"], res["orig_img_w"])
            diff = abs(pred_val - gt_val)
            diffs[t].append(diff)
            
            # Multiple choice accuracy check
            pred_option, _ = closest_option(pred_val, scenario["options_list"])
            if pred_option == gt_option:
                corrects[t] += 1
                
    print("\n--- FINAL RESULTS ACROSS 100 IMAGES ---")
    num_eval = len(diffs[thresholds[0]])
    for t in thresholds:
        mean_diff = sum(diffs[t]) / num_eval if num_eval > 0 else 0
        acc = corrects[t] / num_eval if num_eval > 0 else 0
        print(f"Threshold {t:0.2f} | Mean Absolute Error: {mean_diff:0.3f}% | MCQ Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
