# src/profile_tokenizer.py
import json
import os
from transformers import AutoTokenizer
from normalize import normalize_text

def calculate_fertility(tokenizer, text_list):
    total_tokens = 0
    total_words = 0
    
    for text in text_list:
        clean_text = normalize_text(text)
        words = clean_text.split()
        tokens = tokenizer.encode(clean_text, add_special_tokens=False)
        
        total_words += len(words)
        total_tokens += len(tokens)
        
    fertility = total_tokens / total_words if total_words > 0 else 0
    return fertility, total_tokens, total_words

def profile_dataset(train_file):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training split not found: {train_file}. Run Phase 1 first.")
        
    questions = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            questions.append(entry["code_mixed_question"])
            
    print(f"Loaded {len(questions)} code-mixed questions for profiling.\n")
    
    # Baseline tokenizers to compare
    model_names = [
        "xlm-roberta-base",
        "bert-base-multilingual-cased"
    ]
    
    for model_name in model_names:
        print(f"--- Profiling Tokenizer: {model_name} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            fertility, total_tokens, total_words = calculate_fertility(tokenizer, questions)
            
            print(f"Total Words:     {total_words}")
            print(f"Total Tokens:    {total_tokens}")
            print(f"Token Fertility: {fertility:.2f} tokens/word\n")
        except Exception as e:
            print(f"Could not load {model_name}: {e}\n")

if __name__ == "__main__":
    train_path = os.path.join("data", "splits", "train.jsonl")
    profile_dataset(train_path)