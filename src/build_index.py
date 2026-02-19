# In src/build_index.py
import numpy as np
import faiss
import os

def build_faiss_index(embeddings):
    """Builds a FAISS index from a set of embeddings."""
    # Dimension of the embeddings
    d = embeddings.shape[1]
    
    # Using a simple L2 distance index
    index = faiss.IndexFlatL2(d)
    
    print("Adding embeddings to the FAISS index...")
    index.add(embeddings)
    
    print(f"Index created successfully with {index.ntotal} vectors.")
    return index

def main():
    """Main function to load embeddings and build/save the FAISS index."""
    RESULTS_DIR = "results"
    EMBEDDINGS_PATH = os.path.join(RESULTS_DIR, "embeddings.npy")
    INDEX_PATH = os.path.join(RESULTS_DIR, "image_index.faiss")

    if not os.path.exists(EMBEDDINGS_PATH):
        print(f"Error: Embeddings file not found at {EMBEDDINGS_PATH}")
        print("Please run 'python src/create_embeddings.py' first.")
        return

    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32') # FAISS requires float32
    
    index = build_faiss_index(embeddings)
    
    print(f"Saving FAISS index to {INDEX_PATH}...")
    faiss.write_index(index, INDEX_PATH)
    
    print("Index building complete.")

if __name__ == "__main__":
    main()
