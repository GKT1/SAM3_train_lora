# SAM3 Project Master Directory

This directory contains everything you need to train and evaluate SAM3-LoRA models for disaster damage estimation. It is fully self-contained and optimized for use on powerful cloud GPUs like A100 or H100.

## Directory Structure

- `SAM3_LoRA/`: Contains all training configurations and scripts.
- `sam3/`: The core SAM3 model architecture and utilities, plus new evaluation tools.
- `test_data/`: Contains 100 test scenarios (`sample_RDC.json` and `sample_BDC.json`) and corresponding test images.
- `generate_coco_subset.py`: Utility to generate datasets from raw images.

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r SAM3_LoRA/requirements.txt
   cd sam3 && pip install -e .
   ```

2. **Prepare Data**:
   Upload or generate your COCO formatted dataset and place it in the `disaster_subset_coco` folder (or update the config paths). If you need to generate a new subset:
   ```bash
   python generate_coco_subset.py --num_images 1000
   ```

## Training on a Strong GPU (A100/H100)

If you have a powerful GPU (40GB+ VRAM), you can use the **Full** LoRA configuration, which applies LoRA to the heavy Vision Encoder and increases batch size for much better results.

```bash
cd SAM3_LoRA

# Run training in the background using nohup
nohup python train_sam3_lora_native.py \
  --config configs/full_lora_config.yaml \
  --eval_mqa > training.log 2>&1 &
```

### Advanced Training Controls
- `--eval_mqa`: Enables Multiple Choice Accuracy (MQA) evaluation at the end of each epoch using the `test_data` folder. This gives a much clearer picture of real-world task performance than simple validation loss.
- `--max_train_samples`: Limits the number of training samples for quick iteration (e.g., `--max_train_samples 100`).

## Optimizing the Inference Threshold

Once you have a trained model, finding the exact right `confidence_threshold` is critical. A default of `0.5` might be too high, but `0.1` is often too low, resulting in significant over-segmentation.

We've provided a highly optimized script that tests multiple thresholds across 100 images without needing to re-run the heavy image backbone for each threshold.

```bash
cd sam3
python test_all_thresholds.py
```
*Note: Make sure to update the paths in `test_all_thresholds.py` to point to your new `best_lora_weights.pt` if necessary.*

The output will show you exactly which threshold minimizes the Mean Absolute Error and maximizes MCQ Accuracy.

## Inference

Run inference on single images using your optimized threshold:
```bash
python SAM3_LoRA/infer_sam.py \
  --config SAM3_LoRA/configs/full_lora_config.yaml \
  --weights SAM3_LoRA/outputs/sam3_lora_full/best_lora_weights.pt \
  --image test_data/test_images/your_image.png \
  --prompt "road" \
  --threshold 0.25