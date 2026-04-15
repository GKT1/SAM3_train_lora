import json
import os
import hashlib
from sam3.agent.agent_core import agent_inference

def run_single_image_inference(
    image_paths,
    text_prompt,
    llm_config,
    send_generate_request,
    call_sam_service,
    output_dir="agent_output",
    debug=False,
):
    """Run inference on single or multiple images with provided prompt (FIXED for long filenames)"""

    llm_name = llm_config["name"].replace("/", "_")

    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file names
    # Use the first image for base name if multiple
    image_basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
    if len(image_paths) > 1:
        image_basename += f"_and_{len(image_paths)-1}_others"
    
    # --- FIX: Truncate long prompts for filenames ---
    # 1. Remove dangerous characters like newlines and slashes
    clean_prompt = text_prompt.replace("/", "_").replace(" ", "_").replace("\n", "_").replace(":", "")
    
    # 2. If prompt is longer than 50 chars, cut it and add a hash
    if len(clean_prompt) > 50:
        # Create a short hash of the full prompt so filenames remain unique
        prompt_hash = hashlib.md5(text_prompt.encode()).hexdigest()[:8]
        prompt_for_filename = f"{clean_prompt[:50]}_{prompt_hash}"
    else:
        prompt_for_filename = clean_prompt
    # ------------------------------------------------

    base_filename = f"{image_basename}_{prompt_for_filename}_agent_{llm_name}"
    
    # Final safety clamp on length (Linux limit is 255)
    if len(base_filename) > 200:
        base_filename = base_filename[:200]

    output_json_path = os.path.join(output_dir, f"{base_filename}_pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_history.json")

    # Check if output already exists and skip
    if os.path.exists(output_json_path):
        print(f"Output JSON {output_json_path} already exists. Skipping.")
        return output_image_path

    # Check for existing sam_out folders (skip failed/partial runs)
    sam_output_dir = os.path.join(output_dir, "sam_out")
    if os.path.exists(sam_output_dir):
        for img_path in image_paths:
            # SAM service creates folders with / replaced by -
            expected_folder = img_path.replace("/", "-")
            full_sam_path = os.path.join(sam_output_dir, expected_folder)
            if os.path.exists(full_sam_path):
                print(f"Found existing SAM output folder {full_sam_path}. Skipping sample as per request.")
                return output_image_path

    print(f"{'-' * 30} Starting SAM 3 Agent Session... {'-' * 30} ")
    
    # Pass debug=False to inner function to prevent it from creating its own long filenames
    agent_history, final_output_dict, rendered_final_output = agent_inference(
        image_paths,
        text_prompt,
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        output_dir=output_dir,
        debug=False, # Force False here to avoid inner errors
    )
    print(f"{'-' * 30} End of SAM 3 Agent Session... {'-' * 30} ")

    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_paths"] = image_paths

    # Save outputs
    print(f"Saving to: {output_json_path}")
    json.dump(final_output_dict, open(output_json_path, "w"), indent=4)
    json.dump(agent_history, open(agent_history_path, "w"), indent=4)
    rendered_final_output.save(output_image_path)

    print(f"\n✅ Successfully processed single image!")
    return output_image_path

def run_safe_inference(
    image_paths,
    text_prompt,
    llm_config,
    send_generate_request,
    call_sam_service,
    output_dir="agent_output",
    debug=False,
):
    """
    A SAFE version of inference that guarantees filenames are short and valid.
    Call this instead of run_single_image_inference.
    """
    print(f"🔵 Running SAFE Inference for: {llm_config.get('name', 'Unknown')}")

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Clean the Model Name (remove spaces and slashes)
    llm_name = llm_config["name"].replace(" ", "_").replace("/", "_").replace("\\", "_")

    # 2. Clean the Image Name
    image_basename = os.path.splitext(os.path.basename(image_paths[0]))[0]
    if len(image_paths) > 1:
        image_basename += f"_plus_{len(image_paths)-1}"
    
    # 3. Clean and Truncate the Prompt (Crucial Step)
    # Remove newlines, slashes, and spaces
    clean_prompt = text_prompt.replace("/", "").replace("\\", "").replace(" ", "_").replace("\n", "").replace(":", "")
    
    # Force truncate to 50 characters max
    if len(clean_prompt) > 50:
        # Add a hash so different long prompts don't overwrite each other
        prompt_hash = hashlib.md5(text_prompt.encode()).hexdigest()[:6]
        short_prompt = f"{clean_prompt[:50]}_{prompt_hash}"
    else:
        short_prompt = clean_prompt

    # 4. Construct Safe Filename
    base_filename = f"{image_basename}_{short_prompt}_agent_{llm_name}"
    
    # Double check total length (Max 200 chars)
    if len(base_filename) > 200:
        base_filename = base_filename[:200]

    output_json_path = os.path.join(output_dir, f"{base_filename}_pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_history.json")

    print(f"📝 Saving to safe filename: {os.path.basename(output_json_path)}")

    # Check if output already exists
    if os.path.exists(output_json_path):
        print(f"   Output already exists. Skipping inference.")
        return output_image_path

    print(f"{'-' * 30} Starting SAM 3 Agent Session... {'-' * 30} ")
    
    # Run the agent
    agent_history, final_output_dict, rendered_final_output = agent_inference(
        image_paths,
        text_prompt,
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        output_dir=output_dir,
        debug=debug,
    )
    print(f"{'-' * 30} End of SAM 3 Agent Session... {'-' * 30} ")

    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_paths"] = image_paths

    # Save outputs
    json.dump(final_output_dict, open(output_json_path, "w"), indent=4)
    json.dump(agent_history, open(agent_history_path, "w"), indent=4)
    rendered_final_output.save(output_image_path)

    print(f"\n✅ Success!")
    return output_image_path

print("✅ Fixed function loaded. You can now run the inference cell.")