# In src/create_embeddings.py
import torch
import medmnist
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import numpy as np

def get_model_and_processor(model_name="openai/clip-vit-base-patch32"):
    """Loads the CLIP model and processor."""
    print(f"Loading model: {model_name}...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    print("Model and processor loaded.")
    return model, processor

def get_dataset(split='test'):
    """Loads the PneumoniaMNIST test set."""
    return medmnist.PneumoniaMNIST(split=split, download=True)

def create_embeddings(model, processor, dataset):
    """Generates embeddings for all images in the dataset."""
    model.eval()
    if torch.cuda.is_available():
        model = model.to('cuda')

    embeddings = []
    labels = []
    
    print(f"Generating embeddings for {len(dataset)} images...")
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]
            
            # Preprocess the image. CLIPProcessor expects a list of images.
            inputs = processor(images=[image], return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Get the image embedding
            image_features = model.get_image_features(**inputs)
            
            # Move to CPU and convert to numpy
            embeddings.append(image_features.cpu().numpy().squeeze())
            labels.append(label[0])
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(dataset)} images.")

    return np.array(embeddings), np.array(labels)

def main():
    """Main function to create and save embeddings."""
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
    LABELS_PATH = os.path.join(OUTPUT_DIR, "embedding_labels.npy")

    model, processor = get_model_and_processor()
    dataset = get_dataset('test')
    
    embeddings, labels = create_embeddings(model, processor, dataset)
    
    print(f"Saving embeddings to {EMBEDDINGS_PATH}")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    print(f"Saving labels to {LABELS_PATH}")
    np.save(LABELS_PATH, labels)
    
    print("Embedding creation complete.")

if __name__ == "__main__":
    main()