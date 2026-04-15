import os
import json
import glob
import re

def evaluate_task(task_name, ground_truth_file, output_dir):
    print(f"\n{'='*50}\nEvaluating Task: {task_name}\n{'='*50}")
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return

    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found: {output_dir}")
        return

    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        scenarios = json.load(f)
    
    # Load all predictions
    pred_files = glob.glob(os.path.join(output_dir, "*_pred.json"))
    predictions = {}
    
    for pf in pred_files:
        try:
            with open(pf, 'r') as f:
                data = json.load(f)
                # Key by the basename of the first image path for matching
                if "image_paths" in data and data["image_paths"]:
                    # Normalize path separators and get basename
                    img_name = os.path.basename(data["image_paths"][0].replace("\\", "/"))
                    if img_name not in predictions:
                        predictions[img_name] = []
                    predictions[img_name].append(data)
        except Exception as e:
            print(f"Error loading {pf}: {e}")

    total = len(scenarios)
    correct = 0
    missing = 0
    incomplete = 0
    failed_parsing = 0
    
    sam_out_dir = os.path.join(output_dir, "sam_out")

    print(f"Total scenarios: {total}")
    print(f"Total predictions found: {len(pred_files)}")

    # Track usage of prediction files
    prediction_usage = {}

    for i, scenario in enumerate(scenarios):
        # Match scenario to prediction
        # scenario["pre_image_path"] looks like "test_images\\filename.png"
        
        rel_img_path = scenario["pre_image_path"].replace("\\", "/")
        scenario_img_name = os.path.basename(rel_img_path)
        scenario_prompt = scenario.get("prompts", "")
        
        matched_pred = None
        if scenario_img_name in predictions:
            # Try to match by prompt text
            candidates = predictions[scenario_img_name]
            for cand in candidates:
                # The prediction 'text_prompt' contains question + options.
                # scenario 'prompts' is just the question.
                if scenario_prompt in cand.get("text_prompt", ""):
                    matched_pred = cand
                    break
            
        if not matched_pred:
            # Check if it was attempted (exists in sam_out)
            is_incomplete = False
            if os.path.exists(sam_out_dir):
                for f in os.listdir(sam_out_dir):
                    if f.endswith(scenario_img_name):
                        is_incomplete = True
                        break
            
            if is_incomplete:
                incomplete += 1
            else:
                missing += 1
            continue
            
        pred = matched_pred
        
        # Track usage
        pred_id = id(pred) # simple unique id for the dict object
        if pred_id not in prediction_usage:
            prediction_usage[pred_id] = []
        prediction_usage[pred_id].append(i)
        
        # If there's no ground_truth_option (like in RES), skip text evaluation
        gt_option = scenario.get("ground_truth_option", "").strip().upper()
        if not gt_option:
            continue

        # Predicted Answer
        raw_answer = pred.get("answer", "")
        if raw_answer is None:
            raw_answer = ""
        
        # Parse predicted answer (extract "A", "B", etc. or try to match text)
        # Regex to find the option letter at start or isolated
        # Matches "A.", "A)", "Option A", or just "A" at word boundary
        match = re.search(r'\b([A-E])\b', raw_answer)
        if not match:
             # Try matching at start
             match = re.search(r'^([A-E])\.?', raw_answer)
        
        parsed_answer = match.group(1).upper() if match else None
        
        if not parsed_answer:
            print(f"Failed to parse answer: '{raw_answer}' for {scenario_img_name}")
            failed_parsing += 1
            continue
            
        if parsed_answer == gt_option:
            correct += 1

    processed = total - missing - incomplete
    
    print(f"Processed (Found matching prediction): {processed}")
    print(f"Missing (Not Attempted): {missing}")
    print(f"Incomplete (Attempted but failed/interrupted): {incomplete}")
    print(f"Failed to parse answer: {failed_parsing}")
    print(f"Correct: {correct}")
    
    if processed > len(pred_files):
        print(f"Note: Processed count ({processed}) is higher than predictions found ({len(pred_files)}) because some scenarios share the same image and prompt execution.")
        print("Duplicate usage details:")
        for pid, indices in prediction_usage.items():
            if len(indices) > 1:
                print(f"  Prediction used by scenarios indices: {indices}")
                # Print image name for context
                # We need to find which image corresponds to this prediction
                # (We can find it by looking up one of the scenarios)
                scen = scenarios[indices[0]]
                img = os.path.basename(scen["pre_image_path"])
                print(f"  Image: {img}")

    if processed > 0:
        accuracy = (correct / processed) * 100
        print(f"Accuracy (on successfully processed): {accuracy:.2f}%")
    
    accuracy_total = (correct / total) * 100
    print(f"Accuracy (overall): {accuracy_total:.2f}%")

import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, "100_images_test")
    
    # Use command line argument for output directory if provided
    output_folder_name = "openrouter_output"
    if len(sys.argv) > 1:
        output_folder_name = sys.argv[1]
        
    output_base_dir = os.path.join(base_dir, output_folder_name)
    print(f"Using output directory: {output_base_dir}")
    
    # BDC
    bdc_gt = os.path.join(test_folder, "common_samples", "sample_BDC.json")
    bdc_out = os.path.join(output_base_dir, "sample_BDC")
    if os.path.exists(bdc_out):
        evaluate_task("Building Damage Counting (BDC)", bdc_gt, bdc_out)
    
    # RDC
    rdc_gt = os.path.join(test_folder, "common_samples", "sample_RDC.json")
    rdc_out = os.path.join(output_base_dir, "sample_RDC")
    if os.path.exists(rdc_out):
        evaluate_task("Road Damage Counting (RDC)", rdc_gt, rdc_out)

    # RES
    res_gt = os.path.join(test_folder, "common_samples", "sample_RES.json")
    res_out = os.path.join(output_base_dir, "sample_RES")
    if os.path.exists(res_out):
        evaluate_task("Resolution Options (RES)", res_gt, res_out)

if __name__ == "__main__":
    main()
