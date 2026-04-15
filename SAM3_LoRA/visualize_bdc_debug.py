import os
import yaml
import torch
import sys
import json
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import save_masklet_image

def main():
    config_path = "configs/full_lora_config.yaml"
    weights_path = "outputs/sam3_lora_full/lora_weights_epoch_15.pt"
    bdc_scenarios_path = "../test_data/100_images_test/common_samples/sample_BDC.json"
    images_dir = "../test_data/100_images_test/test_images"
    output_dir = "outputs/sam3_lora_full/bdc_debug_vis"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}!")
        return

    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Building SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=True
    )
    
    print("Applying LoRA...")
    lora_cfg = config["lora"]
    lora_config = LoRAConfig(
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
        apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
        apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
        apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
        apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
        apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
    )
    model = apply_lora_to_model(model, lora_config)
    
    print(f"Loading weights from {weights_path}")
    load_lora_weights(model, weights_path)
    model.to(device)
    model.eval()
    
    threshold = 0.1
    processor = Sam3Processor(model, device=device.type, confidence_threshold=threshold)
    
    with open(bdc_scenarios_path, "r") as f:
        scenarios = json.load(f)
    
    # Select a few samples to visualize
    # Choose samples with different building counts
    indices_to_vis = [0, 1, 4, 9, 14] # beriut_explosion_8 (3), ian_hurricane_748 (124), hurricane_harvey_513 (1), hurricane_matthew_82 (71), santa_rosa_wildfire_332 (14)
    
    for idx in indices_to_vis:
        if idx >= len(scenarios): continue
        scenario = scenarios[idx]
        post_img_name = os.path.basename(scenario["post_image_path"].replace("\\", "/"))
        post_img_path = os.path.join(images_dir, post_img_name)
        
        if not os.path.exists(post_img_path):
            print(f"Image not found: {post_img_path}")
            continue
            
        print(f"Processing sample {idx}: {post_img_name} (GT: {scenario['ground_truth']})")
        
        image = Image.open(post_img_path)
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt="building")
        
        # We skip NMS entirely to see the raw queries.
        # SAM3 object queries are already designed to be largely distinct.
        filtered_masks = state["masks"]
        filtered_scores = state["scores"]
        
        # Prepare outputs for visualization
        vis_outputs = {
            "out_boxes_xywh": [],
            "out_probs": [],
            "out_obj_ids": [],
            "out_binary_masks": []
        }
        
        for i in range(filtered_masks.shape[0]):
            mask = filtered_masks[i, 0].cpu().numpy()
            prob = filtered_scores[i].item()
            vis_outputs["out_binary_masks"].append(mask)
            vis_outputs["out_probs"].append(prob)
            vis_outputs["out_obj_ids"].append(i)
            # Dummy box for vis
            vis_outputs["out_boxes_xywh"].append([0, 0, 0, 0])
            
        out_path = os.path.join(output_dir, f"vis_sample_{idx}_{post_img_name}")
        save_masklet_image(image, vis_outputs, out_path)
        print(f"  Saved visualization to {out_path}")
        print(f"  Number of masks produced: {len(vis_outputs['out_obj_ids'])}")

if __name__ == "__main__":
    main()
