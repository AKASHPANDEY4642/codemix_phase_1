# src/retriever.py
import os
import json
import pickle
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from script_normalization import process_query_pipeline

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = os.path.join("data", "index")

class HybridRetriever:
    def __init__(self):
        print("Loading Full Medical Knowledge Base Indexes...")
        
        # Load Metadata
        metadata_path = os.path.join(INDEX_DIR, "full_kb_metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        # Load Sparse Index
        bm25_path = os.path.join(INDEX_DIR, "bm25_full_index.pkl")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            
        # Load Dense Index
        faiss_path = os.path.join(INDEX_DIR, "dense_full_index.faiss")
        self.faiss_index = faiss.read_index(faiss_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Init Groq for HyDE
        with open("keys.txt", "r") as f:
            api_key = f.read().strip()
        self.groq_client = Groq(api_key=api_key)

    def generate_hyde_document(self, query: str) -> str:
        prompt = f"Provide a brief, factual medical answer in English to this query: {query}"
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return query

    def sparse_search(self, query: str, top_k=20):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

    def dense_search(self, query: str, top_k=20):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_vector, top_k)
        return [(indices[0][i], distances[0][i]) for i in range(len(indices[0])) if indices[0][i] != -1]

    def reciprocal_rank_fusion(self, sparse_results, dense_results, k=60):
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            
        sorted_fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused

    def retrieve(self, raw_query: str, top_k=3):
        norm_data = process_query_pipeline(raw_query)
        medical_terms_query = " ".join(norm_data["medical_entities"])
        if not medical_terms_query:
            medical_terms_query = norm_data["normalized_query"]
            
        hyde_doc = self.generate_hyde_document(norm_data["normalized_query"])
        
        sparse_res = self.sparse_search(medical_terms_query, top_k=20)
        dense_res = self.dense_search(hyde_doc, top_k=20)
        
        fused_res = self.reciprocal_rank_fusion(sparse_res, dense_res, k=60)
        
        retrieved_contexts = []
        for doc_id, score in fused_res[:top_k]:
            chunk_info = self.metadata[doc_id]
            retrieved_contexts.append({
                "chunk_id": doc_id,
                "doc_id": chunk_info["doc_id"],
                "source": chunk_info["source"],
                "text": chunk_info["text"],
                "rrf_score": score
            })
            
        return {
            "query_info": norm_data,
            "hyde_document": hyde_doc,
            "retrieved_context": retrieved_contexts
        }