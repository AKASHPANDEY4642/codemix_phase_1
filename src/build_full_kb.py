# src/build_full_kb.py
import os
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAW_DATA_PATH = os.path.join("data", "raw", "pubmed_qa_raw.json")
INDEX_DIR = os.path.join("data", "index")

def chunk_text_with_metadata(doc_id, text, source, chunk_words=80, overlap_words=15):
    """Splits document text into overlapping chunks while preserving citation metadata."""
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return []
        
    for i in range(0, len(words), chunk_words - overlap_words):
        chunk_words_list = words[i:i + chunk_words]
        chunk_text = " ".join(chunk_words_list)
        
        chunks.append({
            "doc_id": doc_id,
            "source": source,
            "text": chunk_text
        })
        
        if i + chunk_words >= len(words):
            break
            
    return chunks

def build_full_knowledge_base():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}. Run Phase 1 first.")
        
    print(f"Loading raw medical corpus from {RAW_DATA_PATH}...")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    print(f"Loaded {len(raw_data)} raw medical records.")
    
    # 1. Apply Chunking across Corpus
    print("Chunking documents and creating citation metadata...")
    all_chunks = []
    for item in tqdm(raw_data, desc="Chunking Corpus"):
        doc_id = item.get("id", "unknown_id")
        source = item.get("source", "PubMedQA")
        text = f"Question: {item['question']} Answer: {item['answer']}"
        
        item_chunks = chunk_text_with_metadata(doc_id, text, source)
        all_chunks.extend(item_chunks)
        
    print(f"Total semantic chunks created: {len(all_chunks)}")
    
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # Save Metadata Mapping
    metadata_path = os.path.join(INDEX_DIR, "full_kb_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunk metadata to {metadata_path}")
    
    # 2. Build Full Sparse Index (BM25)
    print("Building Full BM25 Sparse Index...")
    corpus_texts = [c["text"] for c in all_chunks]
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_path = os.path.join(INDEX_DIR, "bm25_full_index.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25}, f)
    print(f"Saved BM25 index to {bm25_path}")
    
    # 3. Build Full Dense Index (FAISS)
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Generating dense embeddings in batches (CPU)...")
    batch_size = 128
    embeddings_list = []
    
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding Chunks"):
        batch = corpus_texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        
    all_embeddings = np.vstack(embeddings_list)
    
    dimension = all_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(all_embeddings)
    
    faiss_path = os.path.join(INDEX_DIR, "dense_full_index.faiss")
    faiss.write_index(faiss_index, faiss_path)
    print(f"Successfully built full FAISS index with {faiss_index.ntotal} vectors at {faiss_path}")

if __name__ == "__main__":
    build_full_knowledge_base()