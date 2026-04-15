import os
import json
import torch
import numpy as np
import pycocotools.mask as mask_utils
import yaml
import sys

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_sam3 import sam3_inference

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SAM3_LoRA"))
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

def get_union_area_percentage(pred_masks, orig_h, orig_w):
    if len(pred_masks) == 0: return 0.0
    rle_masks = [{"size": (orig_h, orig_w), "counts": rle} for rle in pred_masks]
    binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
    union_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for m in binary_masks:
        union_mask = np.logical_or(union_mask, m).astype(np.uint8)
    total_area = np.sum(union_mask)
    return (total_area / (orig_h * orig_w)) * 100.0

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
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]
    
    with open("100_images_test/common_samples/sample_RDC.json", "r") as f:
        scenarios = json.load(f)
        
    test_images = scenarios[:5]
    test_images_dir = "100_images_test/test_images"
    
    print("\n--- Testing Thresholds on 5 Images ---")
    for scenario in test_images:
        post_img_name = os.path.basename(scenario["post_image_path"].replace("\\", "/"))
        post_img = os.path.join(test_images_dir, post_img_name)
        gt_str = scenario["ground_truth"].replace('%', '')
        gt_val = float(gt_str)
        
        print(f"\nImage: {post_img_name} | GT Area: {gt_val}%")
        
        for t in thresholds:
            processor = Sam3Processor(model, device=device, confidence_threshold=t)
            outputs = sam3_inference(processor, post_img, "road")
            
            orig_h = outputs["orig_img_h"]
            orig_w = outputs["orig_img_w"]
            pred_masks = outputs["pred_masks"]
            
            pred_val = get_union_area_percentage(pred_masks, orig_h, orig_w)
            diff = abs(pred_val - gt_val)
            print(f"  Threshold {t:.2f} -> Pred: {pred_val:.2f}% | Diff: {diff:.2f}%")

if __name__ == "__main__":
    main()
