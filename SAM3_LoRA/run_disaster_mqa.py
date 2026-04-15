import os
import yaml
import torch
import sys
import json
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# Add sam3 dir to path to find mqa_evaluator module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sam3"))
from mqa_evaluator import evaluate_mqa_on_dataset

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="outputs/sam3_lora_native_fast/lora_weights_epoch_10.pt")
    args = parser.parse_args()
    
    config_path = "configs/full_lora_config.yaml"
    weights_path = args.weights_path
    
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
    
    if os.path.exists(weights_path):
        print(f"Applying LoRA and loading weights from {weights_path}")
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
        load_lora_weights(model, weights_path)
    else:
        print(f"Warning: weights not found at {weights_path}, running base model.")
        
    model.to(device)
    model.eval()
    
    # Run only at a specific threshold, maybe test a few
    thresholds = [0.25, 0.4]
    scenario_path = "../../DisasterM3_Bench/road_mqa_samples.json"
    images_dir = "../../DisasterM3_Bench/test_images"
    
    if not os.path.exists(scenario_path):
        print(f"Scenario not found: {scenario_path}")
        return
        
    all_results = []
    print(f"\nRunning DisasterM3 MQA Evaluation across thresholds: {thresholds}")
    
    for t in thresholds:
        print(f"Testing Threshold: {t}")
        try:
            mqa_results = evaluate_mqa_on_dataset(
                model=model,
                device=device.type,
                scenarios_path=scenario_path,
                images_dir=images_dir,
                threshold=t
            )
            
            res = {
                "threshold": t,
                "accuracy": mqa_results['accuracy'],
                "mae": mqa_results['mae']
            }
            all_results.append(res)
            print(f"  T={t}: Accuracy: {res['accuracy']:.2%}, MAE: {res['mae']:.4f}")
        except Exception as e:
            print(f"  Evaluation failed for threshold {t}: {e}")
            
    # Final Summary Print
    print("\n" + "="*60)
    print(f"FINAL SUMMARY (Roads, MQA) - Weights: {weights_path}")
    print("="*60)
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'MAE':<10}")
    print("-" * 35)
    for res in all_results:
        print(f"{res['threshold']:<10.2f} | {res['accuracy']:<10.2%} | {res['mae']:<10.4f}")
    print("="*60)
    
    # Save to file
    out_name = os.path.basename(weights_path).replace(".pt", "_mqa.json")
    out_dir = os.path.dirname(weights_path)
    if out_dir:
        out_file = os.path.join(out_dir, out_name)
    else:
        out_file = out_name
        
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()