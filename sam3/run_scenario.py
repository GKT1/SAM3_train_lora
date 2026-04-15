import os
import json
import functools
import torch
from sam3.agent.inference import run_single_image_inference
from sam3.agent.client_llm import send_generate_request
from sam3.agent.client_sam3 import call_sam_service
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# LLM Configuration
LLM_PROVIDER = "vertex" 
LLM_MODEL = "gemini-3-flash-preview" 

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

def run_all_scenarios():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "assets/common_samples")
    
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
    api_key = os.environ.get("GOOGLE_API_KEY")
    llm_config = {"name": LLM_MODEL}
    #"sample_BDC.json",
    sample_files = [ "sample_RDC.json", "sample_RES.json"]
    
    for sample_file in sample_files:
        sample_path = os.path.join(samples_dir, sample_file)
        if not os.path.exists(sample_path):
            print(f"Warning: Sample file {sample_path} not found.")
            continue
            
        print(f"\n{'='*50}\nProcessing sample file: {sample_file}\n{'='*50}")
        with open(sample_path, "r") as f:
            scenarios = json.load(f)
            
        for i, scenario_data in enumerate(scenarios):
            print(f"\n--- Processing Scenario {i+1}/{len(scenarios)}: {scenario_data.get('task')} ---")
            
            pre_img = os.path.join(base_dir, "assets", scenario_data["pre_image_path"].replace("\\", "/"))
            post_img = os.path.join(base_dir, "assets", scenario_data["post_image_path"].replace("\\", "/"))
            image_paths = [pre_img, post_img]
            
            # Check if images exist
            missing = [img for img in image_paths if not os.path.exists(img)]
            if missing:
                print(f"Warning: Missing images: {missing}")
                continue

            text_prompt = f"{scenario_data['prompts']} {scenario_data.get('options_str', '')}"
            
            # Mock LLM function for demonstration if no API key
            mock_llm = not api_key
            generation_counter = 0
            def mock_send_generate_request(messages, *args, **kwargs):
                nonlocal generation_counter
                generation_counter += 1
                if generation_counter == 1:
                    return '<tool>{"name": "segment_phrase", "parameters": {"text_prompt": "intact buildings", "image_idx": 1}}</tool>'
                elif generation_counter == 2:
                    # Demonstrate the new 'answer' parameter
                    return f'<tool>{{"name": "select_masks_and_return", "parameters": {{"final_answer_masks": [1], "answer": "{scenario_data.get("ground_truth_option", "A")}"}}}}</tool>'
                return None

            real_send_request = functools.partial(
                send_generate_request, 
                model=llm_config["name"],
                api_key=api_key,
                api_provider=LLM_PROVIDER
            )
            generate_fn = mock_send_generate_request if mock_llm else real_send_request
            
            try:
                output_path = run_single_image_inference(
                    image_paths=image_paths,
                    text_prompt=text_prompt,
                    llm_config=llm_config,
                    send_generate_request=generate_fn,
                    call_sam_service=call_sam_service_with_processor,
                    output_dir=f"agent_output_{sample_file.split('.')[0]}"
                )
                print(f"Scenario completed. Output saved to {output_path}")
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_all_scenarios()
