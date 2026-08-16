# src/evaluate_ragas.py
import os
import json
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from retriever import HybridRetriever
from generator import MedicalGenerator

# Langchain & Ragas imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall
)
from ragas.run_config import RunConfig

def run_ragas_evaluation(sample_size=5):
    print("=" * 80)
    print("PHASE 8: RAGAS AUTOMATED EVALUATION (GROQ OPTIMIZED)")
    print("=" * 80)
    
    # 1. Initialize Pipeline Components
    retriever = HybridRetriever()
    generator = MedicalGenerator()
    
    with open("keys.txt", "r", encoding="utf-8") as f:
        api_keys = [line.strip() for line in f.readlines() if line.strip()]
        api_key = api_keys[0]
        
    # 2. Configure Free LLM and Embeddings
    print("\nConfiguring RAGAS Judge Models (Groq + HuggingFace)...")
    evaluator_llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 3. Load Test Data
    test_file = os.path.join("data", "splits", "test.jsonl")
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))
            
    samples = test_data[:sample_size]
    print(f"Loaded {len(samples)} samples from test set.")
    
    # 4. Generate Predictions
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("\nGenerating pipeline answers for evaluation...")
    for item in tqdm(samples):
        query = item["code_mixed_question"]
        ground_truth = item["original_answer"]
        
        # Run retrieval
        retrieval_res = retriever.retrieve(query, top_k=3)
        contexts = retrieval_res["retrieved_context"]
        context_texts = [ctx["text"] for ctx in contexts]
        
        # Run generation
        answer = generator.generate_answer(query, contexts)
        
        data_dict["question"].append(query)
        data_dict["answer"].append(answer)
        data_dict["contexts"].append(context_texts)
        data_dict["ground_truth"].append(ground_truth)
        
    # 5. Convert to HuggingFace Dataset
    dataset = Dataset.from_dict(data_dict)
    
    # 6. Run Evaluation
    print("\nRunning RAGAS Metrics (Throttled for Groq Free Tier)...")
    metrics = [
        faithfulness,
        context_precision,
        context_recall
    ]
    
    # Throttle requests to prevent Timeouts and 429 Errors
    run_config = RunConfig(max_workers=1, max_retries=5, timeout=60)
    
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )
    
    # 7. Safely Parse and Display Results
    print("\n" + "=" * 80)
    print("FINAL RAGAS SCORES (0.0 to 1.0)")
    print("=" * 80)
    
    # Convert EvaluationResult object to Pandas DataFrame to extract means
    df = results.to_pandas()
    score_columns = [m.name for m in metrics]
    
    for col in score_columns:
        if col in df.columns:
            avg_score = df[col].mean()
            print(f"{col.replace('_', ' ').title():<20} : {avg_score:.4f}")
            
    print("=" * 80)
    print("Exporting detailed results to data/processed/ragas_results.csv...")
    
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    df.to_csv(os.path.join("data", "processed", "ragas_results.csv"), index=False)
    print("Done!")

if __name__ == "__main__":
    run_ragas_evaluation(sample_size=5)