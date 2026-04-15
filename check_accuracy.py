import json
import glob
import os
import re

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

# Load original scenarios
with open("sam3/100_images_test/common_samples/sample_RDC.json", "r") as f:
    scenarios = json.load(f)

test_images_dir = "sam3/100_images_test/test_images"
sam_output_dir = "sam3/100_images_test_output_no_agent_lora/sample_RDC/sam_out"

predictions = []
for i, scenario_data in enumerate(scenarios):
    post_img_name = os.path.basename(scenario_data["post_image_path"].replace("\\", "/"))
    post_img = os.path.join(test_images_dir, post_img_name)
    
    abs_post_img = os.path.abspath(post_img)
    img_folder_name = abs_post_img.replace("/", "-")
    cls_desc = scenario_data.get("cls_description", "")
    text_prompt = "road" if "road" in cls_desc.lower() else ("building" if "building" in cls_desc.lower() else cls_desc.lower().replace(" counting.", ""))
    
    json_path = os.path.join(sam_output_dir, img_folder_name, f"{text_prompt}.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            sam3_outputs = json.load(f)
            
        pred_masks_rle = sam3_outputs.get("pred_masks", [])
        
        import pycocotools.mask as mask_utils
        import numpy as np
        
        orig_h = int(sam3_outputs["orig_img_h"])
        orig_w = int(sam3_outputs["orig_img_w"])
        
        is_percentage = "%" in scenario_data.get("options_str", "")
        if is_percentage:
            if len(pred_masks_rle) > 0:
                rle_masks = [{"size": (orig_h, orig_w), "counts": rle} for rle in pred_masks_rle]
                binary_masks = [mask_utils.decode(rle) for rle in rle_masks]
                union_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                for m in binary_masks:
                    union_mask = np.logical_or(union_mask, m).astype(np.uint8)
                total_area = np.sum(union_mask)
                computed_val = (total_area / (orig_h * orig_w)) * 100.0
            else:
                computed_val = 0.0
        else:
            computed_val = len(pred_masks_rle)
        
        pred_option, pred_value = closest_option(computed_val, scenario_data["options_list"])
        gt_option = scenario_data["ground_truth_option"]
        is_correct = (pred_option == gt_option)
        
        predictions.append({
            "idx": i,
            "image": post_img_name,
            "predicted": computed_val,
            "pred_option": pred_option,
            "gt_option": gt_option,
            "is_correct": is_correct
        })

if len(predictions) > 0:
    correct = sum(1 for p in predictions if p["is_correct"])
    acc = correct / len(predictions)
    print(f"Current Accuracy over {len(predictions)} processed images: {correct}/{len(predictions)} = {acc:.2%}")
    for p in predictions[:5]:
        print(f"Img: {p['image']} | Pred Count: {p['predicted']} -> {p['pred_option']} | GT: {p['gt_option']} | Correct: {p['is_correct']}")
else:
    print("No outputs found to evaluate.")
