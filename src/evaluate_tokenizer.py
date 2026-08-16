# src/evaluate_tokenizer.py
import json
import os
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from normalize import normalize_text

def evaluate_tokenizers(test_file, custom_tokenizer_path):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test split not found: {test_file}")
        
    questions = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            questions.append(normalize_text(entry["code_mixed_question"]))
            
    total_words = sum(len(q.split()) for q in questions)
    print(f"Loaded {len(questions)} test samples ({total_words} total words).\n")
    
    # 1. Evaluate XLM-RoBERTa (Baseline)
    xlm_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    xlm_tokens = sum(len(xlm_tokenizer.encode(q, add_special_tokens=False)) for q in questions)
    xlm_fertility = xlm_tokens / total_words
    
    # 2. Evaluate Custom MedManglish Tokenizer
    custom_tokenizer = Tokenizer.from_file(custom_tokenizer_path)
    custom_tokens = sum(len(custom_tokenizer.encode(q).ids) for q in questions)
    custom_fertility = custom_tokens / total_words
    
    # Results Summary Table
    print("=" * 60)
    print("TOKENIZATION EVALUATION METRICS (PHASE 2)")
    print("=" * 60)
    print(f"{'Tokenizer':<30} | {'Total Tokens':<12} | {'Fertility':<10}")
    print("-" * 60)
    print(f"{'XLM-RoBERTa (Baseline)':<30} | {xlm_tokens:<12} | {xlm_fertility:.2f}")
    print(f"{'MedManglish (Custom BPE)':<30} | {custom_tokens:<12} | {custom_fertility:.2f}")
    print("=" * 60)
    
    reduction = ((xlm_tokens - custom_tokens) / xlm_tokens) * 100
    print(f"\nToken Tax Reduction: {reduction:.2f}% fewer tokens required!")

if __name__ == "__main__":
    test_path = os.path.join("data", "splits", "test.jsonl")
    custom_path = os.path.join("models", "medmanglish_tokenizer", "tokenizer.json")
    evaluate_tokenizers(test_path, custom_path)