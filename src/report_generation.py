"""
This script implements the medical report generation pipeline using a visual language model (VLM).
It loads a pre-trained model (e.g., LLaVA-Med), processes chest X-ray images from the
PneumoniaMNIST dataset, and generates natural language reports based on different prompting strategies.
"""

import torch
import medmnist
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os

def get_model_and_processor(model_name="bczhou/llava-med-v1.5-7b"):
    """
    Loads the LLaVA-Med model and processor from Hugging Face.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and processor.
    """
    print(f"Loading model: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully.")
    return model, processor
    """
    Loads the LLaVA-Med model and processor from Hugging Face.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple: A tuple containing the model and processor.
    """
    print(f"Loading model: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully.")
    return model, processor

def get_dataset(split='test'):
    """
    Downloads and returns the specified split of the PneumoniaMNIST dataset.

    Args:
        split (str): The dataset split to download ('train', 'val', or 'test').

    Returns:
        medmnist.dataset.PneumoniaMNIST: The requested dataset.
    """
    return medmnist.PneumoniaMNIST(split=split, download=True)

def generate_report(model, processor, image, prompt_text):
    """
    Generates a medical report for a given image and prompt.

    Args:
        model: The loaded VLM model.
        processor: The processor for the model.
        image (PIL.Image): The input image.
        prompt_text (str): The user's question for the model.

    Returns:
        str: The generated report.
    """
    # LLaVA-Med requires a specific prompt format
    prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

    # The image needs to be converted to RGB for LLaVA
    rgb_image = image.convert("RGB")

    inputs = processor(text=prompt, images=rgb_image, return_tensors="pt").to(model.device, torch.float16)

    generation_kwargs = dict(
        max_new_tokens=512,
    )

    generate_ids = model.generate(**inputs, **generation_kwargs)

    # Decode the generated ids, skipping the special tokens
    decoded_output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # The output includes the prompt, so we need to remove it.
    # The actual response from the assistant is what comes after "ASSISTANT:".
    assistant_keyword = "ASSISTANT:"
    report_start_index = decoded_output.rfind(assistant_keyword)
    if report_start_index != -1:
        report = decoded_output[report_start_index + len(assistant_keyword):].strip()
    else:
        # Fallback if the keyword is not in the output for some reason
        report = decoded_output

    return report

def main():
    """
    Main function to run the report generation pipeline.
    """
    # --- Configuration ---
    MODEL_NAME = "bczhou/llava-med-v1.5-7b" # Using a public, non-gated model
    NUM_SAMPLES = 10
    OUTPUT_DIR = "results/generated_reports"

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Model and Data ---
    try:
        model, processor = get_model_and_processor(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have sufficient RAM and are logged into Hugging Face if required.")
        return

    dataset = get_dataset('test')

    # --- Define Prompts ---
    prompts = {
        "simple_description": "Describe the findings in this chest X-ray.",
        "pneumonia_focused": "Analyze this chest X-ray for signs of pneumonia. Is pneumonia present?",
        "radiologist_roleplay": "You are a radiologist. Provide a detailed report for this chest X-ray, noting any abnormalities, and conclude with your impression.",
    }

    # --- Generate Reports for a Sample of Images ---
    print(f"Generating reports for {NUM_SAMPLES} images...")

    for i in range(NUM_SAMPLES):
        image, label = dataset[i]
        label_text = "pneumonia" if label[0] == 1 else "normal"

        # Save the original image for reference
        image.save(os.path.join(OUTPUT_DIR, f"image_{i}_{label_text}.png"))

        # Generate report for each prompt
        for prompt_name, prompt_text in prompts.items():
            print(f"  - Generating report for image {i} with prompt '{prompt_name}'...")
            report = generate_report(model, processor, image, prompt_text)

            # Save the report
            report_filename = os.path.join(OUTPUT_DIR, f"report_image_{i}_{label_text}_{prompt_name}.txt")
            with open(report_filename, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"    - Report saved to {report_filename}")

    print("\nReport generation complete.")
    print(f"Generated reports and images are saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()