import os
import json
import functools
import torch
import glob
from sam3.agent.inference import run_single_image_inference
from sam3.agent.client_llm import send_generate_request
from sam3.agent.client_sam3 import call_sam_service
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# LLM Configuration
# Defaults, can be overridden by environment variables
LLM_PROVIDER = "vertex"
LLM_MODEL = "gemini-3-flash-preview"

# Adjust model default if switching to vertex/gemini
if LLM_PROVIDER in ["vertex", "gemini"] and "deepseek" in LLM_MODEL:
    LLM_MODEL = "gemini-3-flash-preview"

SERVER_URL = None
if LLM_PROVIDER == "deepseek":
    SERVER_URL = "https://api.deepseek.com"

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(base_dir, '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print(f"Loaded configuration from {dotenv_path}")
except ImportError:
    pass

def run_100_images_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, "100_images_test")
    samples_dir = os.path.join(test_folder, "common_samples")
    test_images_dir = os.path.join(test_folder, "test_images")
    output_base_dir = os.path.join(base_dir, "100_images_test_output")

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

    call_sam_service_with_processor = functools.partial(call_sam_service, processor)
    
    api_key = None
    if LLM_PROVIDER == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("Warning: DEEPSEEK_API_KEY not found. Please set it in .env or environment variables.")
    elif LLM_PROVIDER in ["vertex", "gemini"]:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found. Please set it in .env or environment variables.")
    else:
        # Generic fallback
        api_key = os.environ.get("OPENAI_API_KEY")

    llm_config = {"name": LLM_MODEL}
    
    # Find all JSON files in samples_dir
    sample_files = glob.glob(os.path.join(samples_dir, "*.json"))
    
    for sample_file_path in sample_files:
        sample_filename = os.path.basename(sample_file_path)
        print(f"\n{'='*50}\nProcessing sample file: {sample_filename}\n{'='*50}")
        
        with open(sample_file_path, "r") as f:
            scenarios = json.load(f)
            
        for i, scenario_data in enumerate(scenarios):
            print(f"\n--- Processing Scenario {i+1}/{len(scenarios)} from {sample_filename}: {scenario_data.get('task')} ---")
            
            # Fix image paths
            # The JSON contains "test_images\\filename.png", we need to convert it
            pre_img_name = os.path.basename(scenario_data["pre_image_path"].replace("\\", "/"))
            post_img_name = os.path.basename(scenario_data["post_image_path"].replace("\\", "/"))
            
            pre_img = os.path.join(test_images_dir, pre_img_name)
            post_img = os.path.join(test_images_dir, post_img_name)
            
            image_paths = [prelà _img, post_img]
            
            # Check if images exist
            missing = [img for img in image_paths if not os.path.exists(img)]
            if missing:
                print(f"Warning: Missing images: {missing}")
                continue

            text_prompt = f"{scenario_data['prompts']} {scenario_data.get('options_str', '')}"
            
            real_send_request = functools.partial(
                send_generate_request, 
                model=llm_config["name"],
                api_key=api_key,
                api_provider=LLM_PROVIDER,
                server_url=SERVER_URL
            )
            
            # Use real request if API key exists, otherwise we can't really run it meaningfully without mocking,
            # but assume user has environment set up or we'll fail.
            # Ideally we should fail if no API key for real inference.
            
            try:
                # Create a subfolder for each sample file output to keep it organized
                sample_output_dir = os.path.join(output_base_dir, sample_filename.split('.')[0])
                if not os.path.exists(sample_output_dir):
                    os.makedirs(sample_output_dir)

                output_path = run_single_image_inference(
                    image_paths=image_paths,
                    text_prompt=text_prompt,
                    llm_config=llm_config,
                    send_generate_request=real_send_request,
                    call_sam_service=call_sam_service_with_processor,
                    output_dir=sample_output_dir
                )
                print(f"Scenario completed. Output saved to {output_path}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_100_images_test()
