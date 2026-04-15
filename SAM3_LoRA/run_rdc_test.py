import os
import yaml
import torch
import sys
import json
from sam3.model_builder import build_sam3_image_model
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

# Add current dir and sam3_core to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_core"))
from mqa_evaluator import evaluate_mqa_on_dataset

def main():
    config_path = "configs/full_lora_config.yaml"
    weights_path = "outputs/sam3_lora_full/lora_weights_epoch_15.pt"
    
    # Original data path
    bench_json_path = "../../DisasterM3_Bench/benchmark_release.json"
    images_dir = "../../DisasterM3_Bench/test_images"
    
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
    
    # Filter benchmark JSON for Road Damage Counting and non-SAR
    print(f"Filtering {bench_json_path} for Road Damage Counting (RDC) tasks...")
    with open(bench_json_path, "r") as f:
        data = json.load(f)
    
    rdc_data = [
        item for item in data 
        if item.get("task") == "Road Damage Counting" and item.get("post_image_type") != "SAR"
    ]
    
    print(f"Found {len(rdc_data)} RDC items (non-SAR).")
    
    # Save to a temporary file for the evaluator
    temp_rdc_path = "rdc_filtered_bench.json"
    with open(temp_rdc_path, "w") as f:
        json.dump(rdc_data, f, indent=4)
        
    threshold = 0.25 # Default threshold
    
    print(f"\nRunning RDC Evaluation (Threshold: {threshold})")
    try:
        results = evaluate_mqa_on_dataset(
            model=model,
            device=device.type,
            scenarios_path=temp_rdc_path,
            images_dir=images_dir,
            threshold=threshold
        )
        
        print("\n" + "="*60)
        print("ROAD DAMAGE COUNTING (RDC) RESULTS")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.2%}")
        print(f"MAE:      {results['mae']:.4f}")
        print("="*60)

        # Save results to file
        output_file = "outputs/sam3_lora_full/rdc_evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
    finally:
        if os.path.exists(temp_rdc_path):
            os.remove(temp_rdc_path)

if __name__ == "__main__":
    main()
