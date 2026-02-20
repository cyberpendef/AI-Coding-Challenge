
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from medmnist import PneumoniaMNIST
from PIL import Image
import os

# Define paths
REPORTS_DIR = 'reports'
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load the dataset
test_dataset = PneumoniaMNIST(split='test', download=True)

# Load the model and processor
processor = AutoProcessor.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
model = AutoModelForCausalLM.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", torch_dtype=torch.bfloat16)

# Select a sample of images (normal, pneumonia, and misclassified)
# In a real scenario, you would have a list of misclassified indices from Task 1
sample_indices = [0, 1, 2, 3, 4, 50, 51, 52, 53, 54]  # Example indices

# Generate reports
report_content = """# Task 2: Medical Report Generation

"""

for i in sample_indices:
    image, label = test_dataset[i]
    label_text = "Pneumonia" if label[0] == 1 else "Normal"

    # Convert to PIL Image
    image = Image.fromarray(image.squeeze(), 'L').convert('RGB')


    # Prepare the prompt
    prompt = "a chest x-ray with"

    # Preprocess the image and prompt
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    # Generate the report
    output = model.generate(**inputs, max_length=256)
    report = processor.decode(output[0], skip_special_tokens=True)

    # Save the image
    image_path = os.path.join(REPORTS_DIR, f"task2_image_{i}.png")
    image.save(image_path)

    # Add to the report content
    report_content += f"""## Image {i} ({label_text})

"""
    report_content += f"""![Image {i}]({os.path.basename(image_path)})

"""
    report_content += f"""**Generated Report:**
{report}

"""

# Write the report to a markdown file
with open(os.path.join(REPORTS_DIR, 'task2_report_generation.md'), 'w') as f:
    f.write(report_content)

print("Task 2 finished. Report and images are saved in the 'reports' directory.")
