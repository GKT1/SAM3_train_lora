import os
import yaml
import torch
import sys
import json
from tqdm import tqdm
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# Add sam3 dir to path to find mqa_evaluator module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../sam3"))
from mqa_evaluator import evaluate_mqa_on_dataset

def main():
    config_path = "configs/full_lora_config.yaml"
    weights_path = "outputs/sam3_lora_full/lora_weights_epoch_15.pt"
    
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
    
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    results_summary = []

    print(f"\nRunning MQA Evaluation across thresholds: {thresholds}")
    
    for t in thresholds:
        print(f"\nTesting Threshold: {t}")
        try:
            mqa_results = evaluate_mqa_on_dataset(
                model=model,
                device=device.type,
                scenarios_path="../test_data/100_images_test/common_samples/sample_BDC.json",
                images_dir="../test_data/100_images_test/test_images",
                threshold=t
            )
            
            res = {
                "threshold": t,
                "accuracy": mqa_results['accuracy'],
                "mae": mqa_results['mae']
            }
            results_summary.append(res)
            
            print(f"Results for T={t}: Accuracy: {res['accuracy']:.2%}, MAE: {res['mae']:.4f}")
        except Exception as e:
            print(f"Evaluation failed for threshold {t}: {e}")

    print("\n" + "="*50)
    print("FINAL SUMMARY (Epoch 15 Weights)")
    print("="*50)
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'MAE':<10}")
    print("-" * 35)
    for res in results_summary:
        print(f"{res['threshold']:<10.2f} | {res['accuracy']:<10.2%} | {res['mae']:<10.4f}")
    print("="*50)

    # Save results to file
    output_file = "outputs/sam3_lora_full/mqa_threshold_results_epoch15.json"
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
