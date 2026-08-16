# src/train_tokenizer.py
import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from normalize import normalize_text

def prepare_corpus(train_file, corpus_file):
    """Extracts text from JSONL to a raw text file for tokenizer training."""
    with open(train_file, "r", encoding="utf-8") as infile, \
         open(corpus_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)
            q_clean = normalize_text(entry.get("code_mixed_question", ""))
            a_clean = normalize_text(entry.get("code_mixed_answer", ""))
            outfile.write(q_clean + "\n")
            outfile.write(a_clean + "\n")

def train_custom_tokenizer(train_file, output_dir, vocab_size=16000):
    corpus_file = os.path.join("data", "processed", "tokenizer_corpus.txt")
    print("Extracting corpus for tokenizer training...")
    prepare_corpus(train_file, corpus_file)
    
    # Initialize a Byte-Pair Encoding (BPE) Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Setup Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2
    )
    
    print(f"Training custom BPE tokenizer (Vocab Size: {vocab_size})...")
    tokenizer.train(files=[corpus_file], trainer=trainer)
    
    # Enable Post-Processing for Transformer Compatibility
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)]
    )
    
    # Save Model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(model_path)
    
    print(f"Custom script-aware tokenizer saved successfully to: {model_path}")

if __name__ == "__main__":
    train_path = os.path.join("data", "splits", "train.jsonl")
    out_dir = os.path.join("models", "medmanglish_tokenizer")
    train_custom_tokenizer(train_path, out_dir, vocab_size=16000)