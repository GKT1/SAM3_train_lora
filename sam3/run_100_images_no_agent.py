import os
import json
import functools
import torch
import glob
import re
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import argparse
import sys
import yaml
from tqdm import tqdm

from sam3.agent.client_sam3 import call_sam_service
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Add SAM3_LoRA to path to import lora_layers
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SAM3_LoRA"))
try:
    from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
except ImportError:
    print("Warning: Could not import lora_layers. Make sure SAM3_LoRA is accessible.")

def parse_option(opt_str):
    """Extract numeric value from option string like 'A) 12%' or '15'."""
    # Remove any non-numeric characters except '.'
    s = str(opt_str).replace('%', '')
    # extract float or int
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    if match:
        return float(match.group())
    return None

def closest_option(computed_val, options_list):
    """Find the closest option to the computed value."""
    closest_idx = 0
    min_diff = float('inf')
    
    for i, opt in enumerate(options_list):
        val = parse_option(opt)
        if val is not None:
            diff = abs(val - computed_val)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
                
    # Usually options_list length is 5: A, B, C, D, E
    return chr(ord('A') + closest_idx), options_list[closest_idx]

def map_cls_description_to_prompt(cls_desc):
    desc = cls_desc.lower().strip()
    if "building" in desc:
        return "building"
    elif "road" in desc:
        return "road"
    return desc.replace(" counting.", "")

def run_100_images_no_agent(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, "100_images_test")
    samples_dir = os.path.join(test_folder, "common_samples")
    test_images_dir = os.path.join(test_folder, "test_images")
    output_base_dir = os.path.join(base_dir, "100_images_test_output_no_agent")
    
    if args.lora_weights:
        output_base_dir += "_lora"

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created output directory: {output_base_dir}")

    # Initialize SAM3 Model
    print("Initializing SAM3 Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            load_from_HF=True,
            enable_inst_interactivity=False
        )
        
        # Apply LoRA if provided
        if args.lora_config and args.lora_weights:
            print(f"Applying LoRA configuration from {args.lora_config}...")
            with open(args.lora_config, 'r') as f:
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
            
            print(f"Loading LoRA weights from {args.lora_weights}...")
            load_lora_weights(model, args.lora_weights)
            
            model.to(device)
            model.eval()
            print("LoRA weights loaded successfully.")
        processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)
        print("SAM3 Model initialized.")
    except Exception as e:
        print(f"Error initializing SAM3 model: {e}")
        return

    call_sam_service_with_processor = functools.partial(call_sam_service, processor)
    
    sample_files = glob.glob(os.path.join(samples_dir, "*.json"))
    
    all_predictions = {}
    
    for sample_file_path in sample_files:
        sample_filename = os.path.basename(sample_file_path)
        print(f"\n{'='*50}\nProcessing sample file: {sample_filename}\n{'='*50}")
        
        sample_output_dir = os.path.join(output_base_dir, sample_filename.split('.')[0])
        os.makedirs(sample_output_dir, exist_ok=True)
        sam_output_dir = os.path.join(sample_output_dir, "sam_out")
        os.makedirs(sam_output_dir, exist_ok=True)
        
        with open(sample_file_path, "r") as f:
            scenarios = json.load(f)
            
        predictions = []
            
        for i, scenario_data in enumerate(tqdm(scenarios, desc=f"Processing {sample_filename}")):
            try:
                # The task might ask about post_image_path or just a generic image
                # From samples, we are asked to analyze the post-disaster image mostly for counting damage/intact.
                # Actually, if we just run SAM3 on the post image, it should be enough for "remaining intact buildings", etc.
                
                post_img_name = os.path.basename(scenario_data["post_image_path"].replace("\\", "/"))
                post_img = os.path.join(test_images_dir, post_img_name)
                
                if not os.path.exists(post_img):
                    print(f"Warning: Missing image: {post_img}")
                    continue
                
                # Determine text prompt
                cls_desc = scenario_data.get("cls_description", "")
                text_prompt = map_cls_description_to_prompt(cls_desc)
                
                # Call SAM3 Service
                path_to_output = call_sam_service_with_processor(
                    image_path=post_img,
                    text_prompt=text_prompt,
                    output_folder_path=sam_output_dir,
                )
                
                # Process SAM3 results
                with open(path_to_output, "r") as f:
                    sam3_outputs = json.load(f)
                    
                pred_masks_rle = sam3_outputs.get("pred_masks", [])
                orig_h = int(sam3_outputs["orig_img_h"])
                orig_w = int(sam3_outputs["orig_img_w"])
                
                is_percentage = "%" in scenario_data.get("options_str", "")
                
                computed_val = 0
                if is_percentage:
                    # Compute union area of all masks
                    if len(pred_masks_rle) > 0:
                        rle_masks = [{"size": (orig_h, orig_w), "counts": rle} for rle in pred_masks_rle]
                        binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
                        # Union
                        union_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        for m in binary_masks:
                            union_mask = np.logical_or(union_mask, m).astype(np.uint8)
                        total_area = np.sum(union_mask)
                        computed_val = (total_area / (orig_h * orig_w)) * 100.0
                    else:
                        computed_val = 0.0
                else:
                    # Just count
                    computed_val = len(pred_masks_rle)
                    
                pred_option, pred_value = closest_option(computed_val, scenario_data["options_list"])
                
                # Record
                pred_entry = {
                    "scenario_idx": i,
                    "image": post_img_name,
                    "prompt_used": text_prompt,
                    "computed_val": computed_val,
                    "predicted_option": pred_option,
                    "predicted_value": pred_value,
                    "ground_truth_option": scenario_data["ground_truth_option"],
                    "ground_truth": scenario_data["ground_truth"],
                    "is_correct": pred_option == scenario_data["ground_truth_option"]
                }
                predictions.append(pred_entry)
                
            except Exception as e:
                print(f"Error on scenario {i}: {e}")
                
        # Calculate accuracy
        correct = sum(1 for p in predictions if p["is_correct"])
        acc = correct / len(predictions) if len(predictions) > 0 else 0
        print(f"\n{sample_filename} Accuracy: {correct}/{len(predictions)} = {acc:.2%}\n")
        
        all_predictions[sample_filename] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(predictions),
            "details": predictions
        }
        
        # Save per-sample results
        out_json_path = os.path.join(sample_output_dir, f"{sample_filename}_predictions.json")
        with open(out_json_path, "w") as f:
            json.dump(all_predictions[sample_filename], f, indent=4)
            
    # Save combined
    final_out_path = os.path.join(output_base_dir, "all_predictions_no_agent.json")
    with open(final_out_path, "w") as f:
        json.dump(all_predictions, f, indent=4)
        
    print(f"Finished. Results saved in {output_base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-config", type=str, help="Path to LoRA config YAML")
    parser.add_argument("--lora-weights", type=str, help="Path to LoRA weights .pt")
    parser.add_argument("--threshold", type=float, default=0.1, help="Confidence threshold for SAM3")
    args = parser.parse_args()
    
    run_100_images_no_agent(args)
