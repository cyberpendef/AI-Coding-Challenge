# In src/search.py
import faiss
import numpy as np
import torch
import medmnist
from transformers import CLIPProcessor, CLIPModel
import argparse
import os
import matplotlib.pyplot as plt

def load_artifacts(results_dir="results", model_name="openai/clip-vit-base-patch32"):
    """Loads all necessary artifacts for searching."""
    print("Loading artifacts...")
    
    # Load FAISS index
    index_path = os.path.join(results_dir, "image_index.faiss")
    index = faiss.read_index(index_path)
    
    # Load embeddings and labels
    embeddings = np.load(os.path.join(results_dir, "embeddings.npy"))
    labels = np.load(os.path.join(results_dir, "embedding_labels.npy"))
    
    # Load dataset
    dataset = medmnist.PneumoniaMNIST(split='test', download=True)
    
    # Load model and processors
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    
    print("Artifacts loaded.")
    return index, embeddings, labels, dataset, model, processor

def visualize_results(query_data, results, labels, dataset, output_path):
    """Saves a visualization of the query and top-k results."""
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(2 * (k + 1), 3))
    
    # Display query
    if isinstance(query_data, str): # Text query
        axes[0].text(0.5, 0.5, f'Query:\n"{query_data}"', ha='center', va='center', wrap=True)
        axes[0].set_title("Text Query")
    else: # Image query
        query_idx, query_label_val = query_data
        query_label = "Pneumonia" if query_label_val == 1 else "Normal"
        axes[0].imshow(dataset[query_idx][0], cmap='gray')
        axes[0].set_title(f"Query: Idx {query_idx}\n({query_label})")
    axes[0].axis('off')

    # Display results
    for i, (res_idx, res_dist) in enumerate(results):
        res_img, res_label_val = dataset[res_idx]
        res_label = "Pneumonia" if res_label_val[0] == 1 else "Normal"
        axes[i+1].imshow(res_img, cmap='gray')
        axes[i+1].set_title(f"Res: Idx {res_idx}\n({res_label})\nD: {res_dist:.2f}")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Results visualization saved to {output_path}")
    plt.close()

def search_image_to_image(index, model, processor, dataset, labels, query_image_idx, k):
    """Performs image-to-image search."""
    query_image, query_label = dataset[query_image_idx]
    
    # Get embedding for the query image
    inputs = processor(images=[query_image], return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs).cpu().numpy()
        
    # Search the index (k+1 because the first result will be the query itself)
    distances, indices = index.search(query_embedding, k + 1)
    
    # Exclude the query image itself from the results
    results = [(idx, dist) for dist, idx in zip(distances[0], indices[0]) if idx != query_image_idx][:k]
    
    return (query_image_idx, query_label[0]), results

def search_text_to_image(index, model, processor, k, query_text):
    """Performs text-to-image search."""
    # Get embedding for the query text
    inputs = processor(text=[query_text], return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).cpu().numpy()
        
    # Search the index
    distances, indices = index.search(query_embedding, k)
    
    results = list(zip(indices[0], distances[0]))
    return query_text, results

def main():
    parser = argparse.ArgumentParser(description="Perform semantic search on medical images.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Image-to-image search parser
    parser_image = subparsers.add_parser("image", help="Image-to-image search")
    parser_image.add_argument("query_idx", type=int, help="Index of the query image in the test set.")
    parser_image.add_argument("-k", type=int, default=5, help="Number of similar images to retrieve.")
    
    # Text-to-image search parser
    parser_text = subparsers.add_parser("text", help="Text-to-image search")
    parser_text.add_argument("query_text", type=str, help="Text description to search for.")
    parser_text.add_argument("-k", type=int, default=5, help="Number of relevant images to retrieve.")
    
    args = parser.parse_args()
    
    # --- Load everything ---
    artifacts = load_artifacts()
    index, embeddings, labels, dataset, model, processor = artifacts
    
    output_dir = "results/retrieval_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Perform search ---
    if args.command == "image":
        print(f"Performing image-to-image search for image index {args.query_idx}...")
        query_data, results = search_image_to_image(index, model, processor, dataset, labels, args.query_idx, args.k)
        output_path = os.path.join(output_dir, f"image_query_{args.query_idx}.png")
        
    elif args.command == "text":
        print(f"Performing text-to-image search for query: '{args.query_text}'...")
        query_data, results = search_text_to_image(index, model, processor, args.k, args.query_text)
        safe_query_text = "".join(c for c in args.query_text if c.isalnum()).lower()
        output_path = os.path.join(output_dir, f"text_query_{safe_query_text}.png")

    # --- Visualize ---
    visualize_results(query_data, results, labels, dataset, output_path)

if __name__ == "__main__":
    main()