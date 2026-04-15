import os
import json
import functools
import torch
import glob
from dotenv import load_dotenv

# Ensure safe inference is used to prevent file name length errors
from sam3.agent.inference import run_safe_inference 
from sam3.agent.client_llm import send_generate_request
from sam3.agent.client_sam3 import call_sam_service
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# LLM Configuration
LLM_PROVIDER = "openrouter"
LLM_MODEL = "qwen/qwen3-vl-8b-instruct"
SERVER_URL = "https://openrouter.ai/api/v1"

# Load environment variables
base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded configuration from {dotenv_path}")

def run_100_images_openrouter():
    test_folder = os.path.join(base_dir, "100_images_test")
    samples_dir = os.path.join(test_folder, "common_samples")
    test_images_dir = os.path.join(test_folder, "test_images")
    output_base_dir = os.path.join(base_dir, "openrouter_100_output")

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
        processor = Sam3Processor(model, device=device)
        print("SAM3 Model initialized.")
    except Exception as e:
        print(f"Error initializing SAM3 model: {e}")
        return

    # Prepare SAM3 execution function
    call_sam_service_wrapper = functools.partial(call_sam_service, processor)
    
    # Get API Key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found. Please set it in .env or environment variables.")
        return

    llm_config = {"name": LLM_MODEL}
    
    # Find all JSON files in samples_dir
    sample_files = glob.glob(os.path.join(samples_dir, "*.json"))
    if not sample_files:
        print(f"No sample JSON files found in {samples_dir}")
        return
    
    for sample_file_path in sample_files:
        sample_filename = os.path.basename(sample_file_path)
        print(f"\n{'='*50}\nProcessing sample file: {sample_filename}\n{'='*50}")
        
        with open(sample_file_path, "r") as f:
            scenarios = json.load(f)
            
        for i, scenario_data in enumerate(scenarios):
            print(f"\n--- Processing Scenario {i+1}/{len(scenarios)} from {sample_filename}: {scenario_data.get('task')} ---")
            
            # Extract Image paths
            pre_img_name = os.path.basename(scenario_data["pre_image_path"].replace("\\", "/"))
            pre_img = os.path.join(test_images_dir, pre_img_name)
            
            # If there's a post image, add it, otherwise just use pre image
            image_paths = [pre_img]
            if "post_image_path" in scenario_data and scenario_data["post_image_path"]:
                post_img_name = os.path.basename(scenario_data["post_image_path"].replace("\\", "/"))
                post_img = os.path.join(test_images_dir, post_img_name)
                image_paths.append(post_img)
            
            # Check if images exist
            missing = [img for img in image_paths if not os.path.exists(img)]
            if missing:
                print(f"Warning: Missing images: {missing}")
                continue

            # Construct Prompt
            prompt_parts = [scenario_data.get('prompts', '')]
            if 'options_str' in scenario_data and scenario_data['options_str']:
                prompt_parts.append(scenario_data['options_str'])
                
            text_prompt = " ".join(prompt_parts)
            
            # Request setup
            real_send_request = functools.partial(
                send_generate_request, 
                model=llm_config["name"],
                api_key=api_key,
                api_provider=LLM_PROVIDER,
                server_url=SERVER_URL
            )
            
            try:
                # Organize by scenario file (e.g., sample_BDC, sample_RDC)
                sample_output_dir = os.path.join(output_base_dir, sample_filename.split('.')[0])
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                # Use run_safe_inference from inference.py 
                output_path = run_safe_inference(
                    image_paths=image_paths,
                    text_prompt=text_prompt,
                    llm_config=llm_config,
                    send_generate_request=real_send_request,
                    call_sam_service=call_sam_service_wrapper,
                    output_dir=sample_output_dir,
                    debug=True # Enable debug to save debug_history.json
                )
                print(f"Scenario completed. Output saved to {output_path}")
            except Exception as e:
                print(f"An error occurred during inference for {sample_filename} (Scenario {i+1}): {e}")

if __name__ == "__main__":
    run_100_images_openrouter()
