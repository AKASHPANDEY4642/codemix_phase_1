# src/indexer.py
import os
import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# We use a fast, lightweight CPU-friendly embedding model for local testing.
# For production/GPU runs later, swap this to 'ncbi/MedCPT-Article-Encoder'
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Sample Medical Knowledge Base (Simulating StatPearls/PubMed abstracts)
SAMPLE_KB = [
    "Hypertension Management: If blood pressure is elevated, the recommended approach includes lifestyle modifications such as reducing sodium intake, regular exercise, and medication like ACE inhibitors or ARBs. Target BP is usually < 130/80 mmHg.",
    "Metformin Side Effects: Metformin is a common medication for type 2 diabetes. Common side effects include gastrointestinal issues like nausea, stomach pain, diarrhea, and vomiting. It should be taken with food to minimize stomach upset.",
    "Fever and Paracetamol: For a viral fever, rest and hydration are key. Paracetamol (acetaminophen) can be taken to reduce fever and relieve headache or body aches. Do not exceed 4000mg per day to avoid liver toxicity.",
    "Myocardial Infarction (Heart Attack): Symptoms include chest pain radiating to the left arm, sweating, and shortness of breath. Immediate emergency care (ICU) and aspirin administration are critical."
]

def chunk_text(text, chunk_size=50, overlap=10):
    """Simple word-based chunker simulating 512-token/50-overlap logic."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def build_indexes(kb_texts, output_dir):
    print("Chunking knowledge base...")
    chunks = []
    for doc in kb_texts:
        chunks.extend(chunk_text(doc, chunk_size=30, overlap=5))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build Sparse Index (BM25)
    print("Building BM25 Sparse Index...")
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(os.path.join(output_dir, "bm25_index.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
        
    # 2. Build Dense Index (FAISS)
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating dense embeddings...")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    faiss.write_index(faiss_index, os.path.join(output_dir, "dense_index.faiss"))
    print(f"Successfully built FAISS index with {faiss_index.ntotal} chunks.")

if __name__ == "__main__":
    index_dir = os.path.join("data", "index")
    build_indexes(SAMPLE_KB, index_dir)