# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import base64
import os
import requests
import json
from typing import Any, Optional

from openai import OpenAI


def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def _prepare_gemini_contents(messages):
    """Helper to prepare contents for Gemini/Vertex requests"""
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_instruction = {"parts": [{"text": msg["content"]}]}
            continue
        
        parts = []
        content = msg.get("content")
        
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image":
                        # Convert image path to base64
                        image_path = item.get("image")
                        try:
                            b64_data, mime_type = get_image_base64_and_mime(image_path)
                            if b64_data:
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": b64_data
                                    }
                                })
                        except Exception as e:
                            print(f"Error processing image for Gemini: {e}")
        
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": parts})
    return contents, system_instruction

def send_vertex_request(messages, model, api_key, thinking_budget=None):
    """
    Sends a request to Google Vertex AI API.
    
    Args:
        messages: List of message objects.
        model: The model ID (e.g., 'gemini-2.5-flash-lite').
        api_key: The Google API Key.
        thinking_budget (int, optional): Token limit for reasoning. 
                                         Pass 0 to disable thinking, -1 for dynamic.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return None

    # Using streamGenerateContent as requested
    url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:streamGenerateContent?key={api_key}"
    
    contents, system_instruction = _prepare_gemini_contents(messages)

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0,
        }
    }
    
    # --- ADDED: Thinking Budget Logic ---
    if thinking_budget is not None:
        payload["generationConfig"]["thinkingConfig"] = {
            "includeThoughts": True, # Set to True to receive "thought" parts in response
            "thinkingBudget": int(thinking_budget)
        }
    # ------------------------------------

    if system_instruction:
        payload["system_instruction"] = system_instruction

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        
        results = response.json() 
        
        full_text = ""
        full_thoughts = "" # Track thoughts separately if needed
        
        for chunk in results:
            if "candidates" in chunk and chunk["candidates"]:
                candidate = chunk["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                # --- UPDATED: Handle Text vs Thoughts ---
                for p in parts:
                    if "text" in p:
                        full_text += p["text"]
                    # If includeThoughts is True, thoughts often come in a 'text' field 
                    # but marked as type 'thought', or in a 'thought' field depending on API version.
                    # Standard Vertex API often puts thoughts in `text` but we can filter if needed.
                    # For now, we append text.
                # ----------------------------------------
                
                finish_reason = candidate.get("finishReason")
                if finish_reason and finish_reason != "STOP":
                     print(f"⚠️ Vertex Finish Reason: {finish_reason}")
        
        return full_text

    except Exception as e:
        print(f"Vertex request failed: {e}")
        if 'response' in locals():
            print(f"Response text: {response.text}")
        return None

def send_gemini_request(messages, model, api_key):
    """Sends a request to Google Gemini API"""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    contents, system_instruction = _prepare_gemini_contents(messages)

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.0, # Low temperature for agentic tasks
        }
    }
    
    if system_instruction:
        payload["system_instruction"] = system_instruction

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Check for safety blocking
        if "promptFeedback" in result:
             feedback = result["promptFeedback"]
             if "blockReason" in feedback:
                 print(f"⚠️ Gemini Prompt Blocked: {feedback['blockReason']}")
                 print(f"Safety Ratings: {feedback.get('safetyRatings', 'N/A')}")

        # Extract text
        if "candidates" in result and result["candidates"]:
            candidate = result["candidates"][0]
            finish_reason = candidate.get("finishReason")
            if finish_reason != "STOP":
                 print(f"⚠️ Gemini Finish Reason: {finish_reason}")
                 if "safetyRatings" in candidate:
                     print(f"Safety Ratings: {candidate['safetyRatings']}")
            
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text_response = "".join([p.get("text", "") for p in parts])
            return text_response
        else:
            print(f"Unexpected Gemini response: {result}")
            return None
            
    except Exception as e:
        print(f"Gemini request failed: {e}")
        if 'response' in locals():
            print(f"Response text: {response.text}")
        return None


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=4096,
    api_provider="gemini",
    token_tracker: Optional[dict] = None,
):
    """
    Sends a request to the OpenAI-compatible API endpoint or Gemini API.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        max_tokens (int): Maximum number of tokens to generate (default: 4096)
        token_tracker (dict, optional): A dictionary to accumulate token usage stats.

    Returns:
        str: The generated response text from the server.
    """
    if api_provider == "vertex":
        print(f"🔍 Calling Vertex AI model {model}...")
        return send_vertex_request(messages, model, api_key)

    if "gemini" in model.lower():
        print(f"🔍 Calling Gemini model {model}...")
        return send_gemini_request(messages, model, api_key)

    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Skip images for DeepSeek as it doesn't support vision
                    if api_provider == "deepseek":
                        print(f"Skipping image for DeepSeek: {c.get('image')}")
                        processed_content.append({"type": "text", "text": "[Image input skipped for DeepSeek model]"})
                        continue

                    # Convert image path to base64 format
                    image_path = c["image"]

                    print("image_path", image_path)
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path
                        )
                        if base64_image is None:
                            print(
                                f"Warning: Could not convert image to base64: {new_image_path}"
                            )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        )

                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key, base_url=server_url)

    try:
        print(f"🔍 Calling model {model}...")
        response = client.chat.completions.create(
            model=model,
            messages=processed_messages,
            max_completion_tokens=max_tokens,
            n=1,
        )
        # print(f"Received response: {response.choices[0].message}")

        # Track token usage if available
        if hasattr(response, "usage") and response.usage:
            print(f"💰 Token Usage: {response.usage}")
            if token_tracker is not None:
                token_tracker["prompt_tokens"] = token_tracker.get("prompt_tokens", 0) + response.usage.prompt_tokens
                token_tracker["completion_tokens"] = token_tracker.get("completion_tokens", 0) + response.usage.completion_tokens
                token_tracker["total_tokens"] = token_tracker.get("total_tokens", 0) + response.usage.total_tokens

        # Extract the response content
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            print(f"Unexpected response format: {response}")
            return None

    except Exception as e:
        print(f"Request failed: {e}")
        return None


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None
