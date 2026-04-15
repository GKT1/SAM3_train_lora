import os
import json
import glob
import re
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image as PILImage
import numpy as np

# Configuration
SELECTION_MODE = "substring" # "random", "substring", or "first"
# Substrings for specific selection (used if SELECTION_MODE is "substring")
SUBSTRINGS = {
    "BDC_CORRECT": "hurricane_michael",
    "BDC_INCORRECT": "moore_tornado",
    "BDC_CORRECT": "hurricane_michael",
    "RDC_INCORRECT": "turkey_earthquake5_pre_336"
}

MASK_FOLDER_MAPPING = {
    "Intact building counting.": "test_building_intact_mask",
    "Damaged building counting.": "test_building_damaged_mask",
    "Totally-destroyed building counting.": "test_building_destroyed_mask",
    "intact road counting.": "test_road_intact_mask",
    "flooding road counting.": "test_road_flooded_mask",
    "Segment flooding areas.": "flooding_mask",
    "Segment lava areas.": "volcano_lava"
}

def load_predictions_with_paths(output_dir):
    pred_files = glob.glob(os.path.join(output_dir, "*_pred.json"))
    predictions = {}
    for pf in pred_files:
        try:
            with open(pf, 'r') as f:
                data = json.load(f)
                data["_json_path"] = pf
                data["_png_path"] = pf.replace("_pred.json", "_pred.png")
                if "image_paths" in data and data["image_paths"]:
                    img_name = os.path.basename(
                        data["image_paths"][0].replace("\\", "/")
                    )
                    if img_name not in predictions:
                        predictions[img_name] = []
                    predictions[img_name].append(data)
        except Exception:
            pass
    return predictions


def load_and_resize(img_path, target_h=512, target_w=512):
    if not img_path or not os.path.exists(img_path):
        # Return blank
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 200
    try:
        img = PILImage.open(img_path).convert("RGB")
        # Resize keeping aspect ratio for width? Or fixed?
        # Fixed width ensures alignment in strip.
        img = img.resize((target_w, target_h), PILImage.LANCZOS)
        return np.array(img)
    except:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 200

def get_gt_mask_path(scenario, masks_root):
    cls_desc = scenario.get("cls_description", "")
    folder_name = MASK_FOLDER_MAPPING.get(cls_desc)
    
    if not folder_name:
        # Try fuzzy match?
        for k, v in MASK_FOLDER_MAPPING.items():
            if k.lower() in cls_desc.lower():
                folder_name = v
                break
    
    if not folder_name:
        return None
        
    # Construct filename from post_image_path
    # Handle potentially un-split path due to backslashes
    clean_path = scenario["post_image_path"].replace("test_images\\", "").replace("test_images/", "")
    base_name = os.path.basename(clean_path)
    name_part = os.path.splitext(base_name)[0]
    
    # Heuristics to strip suffix
    if "_post_disaster" in name_part:
        clean_name = name_part.replace("_post_disaster", "")
    elif "_post_" in name_part:
        # turkey_earthquake5_post_336 -> turkey_earthquake5_336
        clean_name = name_part.replace("_post_", "_")
    else:
        clean_name = name_part

    # Search in folder
    folder_path = os.path.join(masks_root, folder_name)
    if not os.path.exists(folder_path):
        return None
        
    # Direct try
    cand1 = os.path.join(folder_path, clean_name + ".png")
    if os.path.exists(cand1): return cand1
    
    # Try fuzzy find if name cleaning failed
    # e.g. look for file starting with name_part or containing it?
    # This might be slow.
    return None


TASK_FULL_NAMES = {
    "BDC": "Building Damage Counting",
    "RDC": "Road Damage Counting",
}


def create_visualization(task_abbr, status, scenario, pred, gt_mask_path, output_filename):
    color_task = "#83b367"
    color_img_name = "#6c8ebe"
    color_qa = "#d5e8d4"

    # Load and concatenate 4 images into one strip
    h, w = 512, 512
    img_pre = load_and_resize(scenario.get("full_pre_path"), h, w)
    img_post = load_and_resize(scenario.get("full_post_path"), h, w)
    img_res = load_and_resize(pred.get("_png_path"), h, w)
    img_gt = load_and_resize(gt_mask_path, h, w)
    
    strip = np.concatenate([img_pre, img_post, img_res, img_gt], axis=1)

    total_w = strip.shape[1]

    # Add thin label bar on top of image strip
    labels = ["Pre-Disaster", "Post-Disaster", "Agent Result", "GT Mask"]
    label_h = 30
    label_bar = np.ones((label_h, total_w, 3), dtype=np.uint8) * 40  # dark bg
    strip_with_labels = np.concatenate([label_bar, strip], axis=0)

    # Figure with no margins
    fig_w = 20
    img_aspect = strip_with_labels.shape[0] / strip_with_labels.shape[1]
    task_bar_h = 0.6
    qa_bar_h = 2.0
    img_h = fig_w * img_aspect
    fig_h = img_h + task_bar_h + qa_bar_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[img_h, task_bar_h, qa_bar_h],
                  hspace=0, left=0, right=1, top=1, bottom=0)

    # ── Row 0: Image strip ──────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(strip_with_labels)
    ax_img.axis("off")

    # Draw label text on the dark bar
    for i, label in enumerate(labels):
        cx = w * i + w // 2
        ax_img.text(cx, label_h // 2, label,
                    fontsize=13, fontweight="bold", color="white",
                    ha="center", va="center")

    # Draw thin white separator lines between images
    for i in range(1, 4):
        x = w * i
        ax_img.axvline(x=x, color="white", linewidth=2,
                       ymin=0, ymax=1)

    # ── Row 1: Task name full-width bar ─────────────────────────────
    ax_task = fig.add_subplot(gs[1])
    ax_task.axis("off")
    ax_task.set_xlim(0, 1)
    ax_task.set_ylim(0, 1)
    task_full = TASK_FULL_NAMES.get(task_abbr, task_abbr)
    ax_task.add_patch(plt.Rectangle((0, 0), 1, 1,
                      facecolor=color_task, alpha=0.7,
                      transform=ax_task.transAxes, zorder=0))
    
    img_name_str = os.path.basename(scenario['pre_image_path'])
    title_text = f"{task_full} | Status: {status} | Image: {img_name_str}"
    
    ax_task.text(0.5, 0.5, title_text,
                 fontsize=20, fontweight="bold", color="white",
                 ha="center", va="center",
                 transform=ax_task.transAxes)

    # ── Row 2: Q&A full-width bar ───────────────────────────────────
    ax_qa = fig.add_subplot(gs[2])
    ax_qa.axis("off")
    ax_qa.set_xlim(0, 1)
    ax_qa.set_ylim(0, 1)
    ax_qa.add_patch(plt.Rectangle((0, 0), 1, 1,
                    facecolor=color_qa, alpha=0.5,
                    transform=ax_qa.transAxes, zorder=0))

    gt_option = scenario.get("ground_truth_option", "")
    pred_answer = pred.get("answer", "N/A")
    prompt = scenario.get("prompts", "")
    options = scenario.get("options_str", "")

    qa_text = (
        f"Question:      {prompt}\n"
        f"Options:       {options}\n"
        f"Ground Truth:  {gt_option}\n"
        f"Agent Answer:  {pred_answer}"
    )

    ax_qa.text(0.03, 0.5, qa_text,
               fontsize=23, ha="left", va="center",
               family="monospace",
               transform=ax_qa.transAxes)

    os.makedirs(os.path.dirname(output_filename) or ".", exist_ok=True)
    plt.savefig(output_filename, dpi=150, facecolor="white",
                bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Generated: {output_filename}")

def select_example(candidates, key):
    if not candidates: return None
    
    target_substring = SUBSTRINGS.get(key, "")
    filtered = candidates
    
    if SELECTION_MODE == "substring" and target_substring:
        filtered = [c for c in candidates if target_substring in c[0]["pre_image_path"]]
        if not filtered:
            print(f"No candidates match substring '{target_substring}' for {key}")
            return None
            
    if SELECTION_MODE == "random":
        return random.choice(filtered)
    return filtered[0]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, "100_images_test")
    output_base_dir = os.path.join(base_dir, "100_images_test_output")
    masks_root = os.path.join(base_dir, "masks")

    tasks = [
        ("BDC", "sample_BDC.json", "sample_BDC"),
        ("RDC", "sample_RDC.json", "sample_RDC"),
    ]

    for task_abbr, json_file, out_subdir in tasks:
        gt_file = os.path.join(test_folder, "common_samples", json_file)
        out_dir = os.path.join(output_base_dir, out_subdir)

        with open(gt_file, "r") as f:
            scenarios = json.load(f)

        predictions = load_predictions_with_paths(out_dir)

        correct_candidates = []
        incorrect_candidates = []

        for scenario in scenarios:
            test_images_dir = os.path.join(test_folder, "test_images")
            pre_name = os.path.basename(scenario["pre_image_path"].replace("\\", "/"))
            post_name = os.path.basename(scenario["post_image_path"].replace("\\", "/"))
            scenario["full_pre_path"] = os.path.join(test_images_dir, pre_name)
            scenario["full_post_path"] = os.path.join(test_images_dir, post_name)

            scenario_img_name = pre_name
            scenario_prompt = scenario.get("prompts", "")

            matched_pred = None
            if scenario_img_name in predictions:
                for cand in predictions[scenario_img_name]:
                    if scenario_prompt in cand.get("text_prompt", ""):
                        matched_pred = cand
                        break
            if not matched_pred:
                continue

            gt_option = scenario.get("ground_truth_option", "").strip().upper()
            raw_answer = matched_pred.get("answer", "") or ""

            match = re.search(r"\b([A-E])\b", raw_answer)
            if not match:
                match = re.search(r"^([A-E])\.?", raw_answer)
            parsed = match.group(1).upper() if match else None

            if parsed == gt_option:
                correct_candidates.append((scenario, matched_pred))
            elif parsed:
                incorrect_candidates.append((scenario, matched_pred))

        correct_ex = select_example(correct_candidates, f"{task_abbr}_CORRECT")
        incorrect_ex = select_example(incorrect_candidates, f"{task_abbr}_INCORRECT")

        if correct_ex:
            gt_mask = get_gt_mask_path(correct_ex[0], masks_root)
            create_visualization(
                task_abbr, "Correct",
                correct_ex[0], correct_ex[1], gt_mask,
                f"sam3/report_{task_abbr}_correct.png",
            )
        else:
            print(f"No correct example found for {task_abbr}")

        if incorrect_ex:
            gt_mask = get_gt_mask_path(incorrect_ex[0], masks_root)
            create_visualization(
                task_abbr, "Incorrect",
                incorrect_ex[0], incorrect_ex[1], gt_mask,
                f"sam3/report_{task_abbr}_incorrect.png",
            )
        else:
            print(f"No incorrect example found for {task_abbr}")


if __name__ == "__main__":
    main()
