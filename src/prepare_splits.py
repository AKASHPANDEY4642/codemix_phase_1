# import os
# import json
# from pathlib import Path
# from sklearn.model_selection import train_test_split

# ROOT = Path(__file__).resolve().parents[1]

# def save_jsonl(data, file_path):
#     with open(file_path, "w", encoding="utf-8") as f:
#         for entry in data:
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# def split_dataset(input_file, output_dir):
#     if not os.path.exists(input_file):
#         raise FileNotFoundError(f"Input file not found: {input_file}")

#     with open(input_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     if len(data) == 0:
#         print("Dataset is empty. Run generate_codemix.py first.")
#         return

#     # 70% Train, 30% Temp (Val + Test)
#     train, temp = train_test_split(data, test_size=0.30, random_state=42)
#     # Split the 30% evenly into 15% Val and 15% Test
#     val, test = train_test_split(temp, test_size=0.50, random_state=42)

#     os.makedirs(output_dir, exist_ok=True)

#     save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
#     save_jsonl(val, os.path.join(output_dir, "val.jsonl"))
#     save_jsonl(test, os.path.join(output_dir, "test.jsonl"))

#     print("Dataset splitting complete:")
#     print(f" Train: {len(train)} samples (70%) -> {os.path.join(output_dir, 'train.jsonl')}")
#     print(f" Val:   {len(val)} samples (15%) -> {os.path.join(output_dir, 'val.jsonl')}")
#     print(f" Test:  {len(test)} samples (15%) -> {os.path.join(output_dir, 'test.jsonl')}")

# if __name__ == "__main__":
#     processed_file = os.path.join(ROOT, "data", "processed", "synthetic_codemix_qa.json")
#     splits_dir = os.path.join(ROOT, "data", "splits")
#     split_dataset(processed_file, splits_dir)
# src/prepare_splits.py
import os
import json
from sklearn.model_selection import train_test_split

def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def split_dataset(input_file, output_dir):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed file not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} generated QA samples.")

    # 70% Train, 30% Temp (Val + Test)
    train, temp = train_test_split(data, test_size=0.30, random_state=42)
    # Split 30% evenly into 15% Val and 15% Test
    val, test = train_test_split(temp, test_size=0.50, random_state=42)

    os.makedirs(output_dir, exist_ok=True)

    save_jsonl(train, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test, os.path.join(output_dir, "test.jsonl"))

    print("\nSplits generated successfully:")
    print(f"- Train: {len(train)} samples (70%) -> {os.path.join(output_dir, 'train.jsonl')}")
    print(f"- Val:   {len(val)} samples (15%) -> {os.path.join(output_dir, 'val.jsonl')}")
    print(f"- Test:  {len(test)} samples (15%) -> {os.path.join(output_dir, 'test.jsonl')}")

if __name__ == "__main__":
    processed_file = os.path.join("data", "processed", "synthetic_codemix_qa.json")
    splits_dir = os.path.join("data", "splits")
    split_dataset(processed_file, splits_dir)