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

# LLM Configuration for OpenRouter
LLM_PROVIDER = "openrouter"

# User requested model: "qwen/qwen3-vl-8b-instruct"
# Note: As of now, Qwen3 might not be widely available. 
# Common alternatives on OpenRouter are "qwen/qwen-2-vl-7b-instruct" or "qwen/qwen-2.5-vl-72b-instruct".
# You can change this string to the exact model ID you want to use.
LLM_MODEL = "qwen/qwen3-vl-8b-instruct" 

SERVER_URL = "https://openrouter.ai/api/v1"

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

def run_openrouter_test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "assets/common_samples")
    output_base_dir = os.path.join(base_dir, "openrouter_output")

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
    
    # API Key from environment variable
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Warning: OPENROUTER_API_KEY not found. Please set it in .env or environment variables.")
        # Fallback to OpenAI key if available, as OpenRouter is compatible? No, different service.
        return

    llm_config = {"name": LLM_MODEL}
    
    sample_files = ["sample_RDC.json", "sample_RES.json"]
    
    for sample_file in sample_files:
        sample_path = os.path.join(samples_dir, sample_file)
        if not os.path.exists(sample_path):
            print(f"Warning: Sample file {sample_path} not found.")
            continue
            
        print(f"\n{'='*50}\nProcessing sample file: {sample_file}\n{'='*50}")
        
        with open(sample_path, "r") as f:
            scenarios = json.load(f)
            
        for i, scenario_data in enumerate(scenarios):
            print(f"\n--- Processing Scenario {i+1}/{len(scenarios)} from {sample_file}: {scenario_data.get('task')} ---")
            
            pre_img = os.path.join(base_dir, "assets", scenario_data["pre_image_path"].replace("\\", "/"))
            post_img = os.path.join(base_dir, "assets", scenario_data["post_image_path"].replace("\\", "/"))
            image_paths = [pre_img, post_img]
            
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
            
            try:
                # Create a subfolder for each sample file output to keep it organized
                sample_output_dir = os.path.join(output_base_dir, sample_file.split('.')[0])
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
    run_openrouter_test()
