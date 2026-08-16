# import os
# import json
# from pathlib import Path
# from datasets import load_dataset

# ROOT = Path(__file__).resolve().parents[1]

# def download_datasets():
#     print("Downloading PubMedQA raw dataset...")
#     # Using the labeled train split
#     pubmed_qa = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
#     raw_dir = os.path.join(ROOT, "data", "raw")
#     os.makedirs(raw_dir, exist_ok=True)
    
#     pubmed_data = []
#     for item in pubmed_qa:
#         pubmed_data.append({
#             "id": item["pubid"],
#             "question": item["question"],
#             "answer": item["long_answer"],
#             "source": "PubMedQA"
#         })
        
#     output_path = os.path.join(raw_dir, "pubmed_qa_raw.json")
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(pubmed_data, f, ensure_ascii=False, indent=2)
        
#     print(f"Successfully saved {len(pubmed_data)} raw PubMedQA pairs to {output_path}")

# if __name__ == "__main__":
#     download_datasets()
# src/download_data.py
import os
import json
from datasets import load_dataset

def download_datasets(target_count=5500):
    print("Downloading PubMedQA datasets (labeled + artificial)...")
    
    # 1. Load labeled dataset (1,000 samples)
    pqa_labeled = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    pubmed_data = []
    for item in pqa_labeled:
        pubmed_data.append({
            "id": f"labeled_{item['pubid']}",
            "question": item["question"],
            "answer": item["long_answer"],
            "source": "PubMedQA-Labeled"
        })
        
    # 2. Load artificial dataset to reach 5,500
    pqa_artificial = load_dataset("pubmed_qa", "pqa_artificial", split="train")
    for item in pqa_artificial:
        pubmed_data.append({
            "id": f"art_{item['pubid']}",
            "question": item["question"],
            "answer": item["long_answer"],
            "source": "PubMedQA-Artificial"
        })
        if len(pubmed_data) >= target_count:
            break
            
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "pubmed_qa_raw.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pubmed_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(pubmed_data)} raw PubMedQA pairs to {output_path}")

if __name__ == "__main__":
    download_datasets(target_count=5500)