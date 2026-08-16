# import os
# import json
# import time
# from pathlib import Path
# from tqdm import tqdm
# from groq import Groq

# ROOT = Path(__file__).resolve().parents[1]
# KEYS_FILE = ROOT / "keys.txt"

# # --- Configuration ---
# MODEL_NAME = "llama-3.1-8b-instant"  # Higher free-tier limit: 14,400 req/day vs 1,000 for 70b
# REQUEST_DELAY = 2.5  # seconds between successful requests

# # --- Helper: read key from file (allows hot-swapping keys) ---
# def load_api_key():
#     if not os.path.exists(KEYS_FILE):
#         raise FileNotFoundError(f"Please create '{KEYS_FILE}' with your Groq API key.")
#     with open(KEYS_FILE, "r", encoding="utf-8") as f:
#         key = f.read().strip()
#     if not key:
#         raise ValueError(f"No key found in '{KEYS_FILE}'.")
#     return key

# # --- Prompt Template ---
# PROMPT_TEMPLATE = """You are a bilingual Marathi-English speaker visiting a doctor.
# Rewrite the following medical question and answer into natural code-mixed Marathi-English (Manglish).

# Rules:
# 1. Preserve medical terminology in English (e.g., "blood pressure", "metformin", "side effects").
# 2. Conversational parts should use Marathi (in Devanagari script).
# 3. The sentence structure must accurately reflect natural Mumbai/Pune conversational style.

# Input Question: {question}
# Input Answer: {answer}

# Provide your output strictly in JSON format as follows:
# {{
#   "code_mixed_question": "...",
#   "code_mixed_answer": "..."
# }}

# Output ONLY the JSON, no other text."""

# def save_dataset(output_file, dataset):
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(dataset, f, ensure_ascii=False, indent=2)

# def generate_synthetic_data(input_file, output_file, limit=1000):
#     api_key = load_api_key()
#     client = Groq(api_key=api_key)
#     current_key_suffix = api_key[-6:]
#     print(f"Loaded Groq API key: ...{current_key_suffix}")

#     with open(input_file, "r", encoding="utf-8") as f:
#         raw_data = json.load(f)

#     # Resume support
#     generated_dataset = []
#     existing_ids = set()
#     if os.path.exists(output_file):
#         try:
#             with open(output_file, "r", encoding="utf-8") as f:
#                 generated_dataset = json.load(f)
#                 existing_ids = {item["original_id"] for item in generated_dataset}
#             print(f"Resuming from {len(generated_dataset)} saved samples.")
#         except Exception as e:
#             print(f"Could not load existing output ({e}). Starting fresh.")
#             generated_dataset = []

#     remaining = limit - len(generated_dataset)
#     print(f"Target: {limit} samples | Already done: {len(generated_dataset)} | Remaining: {remaining}")
#     print(f"Model: {MODEL_NAME} | Delay: {REQUEST_DELAY}s/request")
#     print(f"Estimated time: ~{remaining * (REQUEST_DELAY + 1) / 60:.0f} minutes\n")

#     pbar = tqdm(total=min(limit, len(raw_data)), desc="Generating Code-Mixed Data", initial=len(generated_dataset))

#     consecutive_errors = 0
#     save_every = 5

#     for item in raw_data:
#         if len(generated_dataset) >= limit:
#             print(f"\nReached target of {limit} samples!")
#             break

#         if consecutive_errors >= 15:
#             tqdm.write(f"\n[STOPPED] 15 consecutive errors. Saving progress and stopping.")
#             break

#         if item["id"] in existing_ids:
#             continue

#         prompt = PROMPT_TEMPLATE.format(
#             question=item["question"],
#             answer=item["answer"]
#         )

#         success = False
#         max_retries = 5

#         for attempt in range(max_retries):
#             try:
#                 response = client.chat.completions.create(
#                     model=MODEL_NAME,
#                     messages=[
#                         {"role": "system", "content": "You output valid JSON only. No markdown, no code blocks."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     temperature=0.7,
#                     max_tokens=1024,
#                     response_format={"type": "json_object"}
#                 )

#                 raw_text = response.choices[0].message.content.strip()

#                 # Clean markdown wrapping if present
#                 if raw_text.startswith("```json"):
#                     raw_text = raw_text[7:]
#                 if raw_text.startswith("```"):
#                     raw_text = raw_text[3:]
#                 if raw_text.endswith("```"):
#                     raw_text = raw_text[:-3]
#                 raw_text = raw_text.strip()

#                 res_json = json.loads(raw_text)

#                 generated_dataset.append({
#                     "original_id": item["id"],
#                     "original_question": item["question"],
#                     "original_answer": item["answer"],
#                     "code_mixed_question": res_json.get("code_mixed_question", ""),
#                     "code_mixed_answer": res_json.get("code_mixed_answer", ""),
#                     "source": item.get("source", "PubMedQA")
#                 })
#                 existing_ids.add(item["id"])
#                 consecutive_errors = 0
#                 success = True
#                 pbar.update(1)

#                 # Periodic save
#                 if len(generated_dataset) % save_every == 0:
#                     save_dataset(output_file, generated_dataset)

#                 time.sleep(REQUEST_DELAY)
#                 break  # Success

#             except Exception as e:
#                 err_msg = str(e)

#                 if "429" in err_msg or "rate_limit" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg:
#                     # Rate limited — wait 60s (Groq resets per-minute quotas every 60s)
#                     wait = 65
#                     tqdm.write(f"[RATE LIMIT] Attempt {attempt+1}/{max_retries}. Waiting {wait}s for quota reset...")
#                     time.sleep(wait)

#                     # Re-read key from file in case user swapped it during the wait
#                     try:
#                         new_key = load_api_key()
#                         if new_key[-6:] != current_key_suffix:
#                             client = Groq(api_key=new_key)
#                             current_key_suffix = new_key[-6:]
#                             tqdm.write(f"[KEY SWAPPED] Now using key: ...{current_key_suffix}")
#                     except Exception:
#                         pass  # Keep using current key

#                 elif "401" in err_msg or "invalid_api_key" in err_msg.lower():
#                     tqdm.write(f"[AUTH ERROR] Key ...{current_key_suffix} is invalid.")
#                     tqdm.write(f"[AUTH ERROR] Update keys.txt with a valid key. Waiting 30s for you to swap...")
#                     time.sleep(30)
#                     # Try to reload key
#                     try:
#                         new_key = load_api_key()
#                         if new_key[-6:] != current_key_suffix:
#                             client = Groq(api_key=new_key)
#                             current_key_suffix = new_key[-6:]
#                             tqdm.write(f"[KEY SWAPPED] Now using key: ...{current_key_suffix}")
#                         else:
#                             tqdm.write(f"[AUTH ERROR] Same invalid key still in file. Stopping.")
#                             consecutive_errors = 15  # Force stop
#                             break
#                     except Exception:
#                         consecutive_errors = 15
#                         break

#                 else:
#                     tqdm.write(f"[ERROR] Item {item['id']}: {err_msg[:150]}")
#                     consecutive_errors += 1
#                     time.sleep(5)
#                     break  # Don't retry parse/other errors

#         if not success:
#             consecutive_errors += 1

#     pbar.close()
#     save_dataset(output_file, generated_dataset)
#     print(f"\nDone! {len(generated_dataset)} total samples saved to {output_file}")

# if __name__ == "__main__":
#     raw_path = os.path.join(ROOT, "data", "raw", "pubmed_qa_raw.json")
#     out_path = os.path.join(ROOT, "data", "processed", "synthetic_codemix_qa.json")
#     generate_synthetic_data(raw_path, out_path, limit=1000)
# src/generate_codemix.py
import os
import json
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq

ROOT = Path(__file__).resolve().parents[1]
KEYS_FILE = ROOT / "keys.txt"

# --- Configuration ---
MODEL_NAME = "llama-3.1-8b-instant"  
REQUEST_DELAY = 2.5  

# --- Helper: read key from file (allows hot-swapping keys) ---
def load_api_key():
    if not os.path.exists(KEYS_FILE):
        raise FileNotFoundError(f"Please create '{KEYS_FILE}' with your Groq API key.")
    with open(KEYS_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise ValueError(f"No key found in '{KEYS_FILE}'.")
    return key

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a bilingual Marathi-English speaker visiting a doctor.
Rewrite the following medical question and answer into natural code-mixed Marathi-English (Manglish).

Rules:
1. Preserve medical terminology in English (e.g., "blood pressure", "metformin", "side effects").
2. Conversational parts should use Marathi (in Devanagari script).
3. The sentence structure must accurately reflect natural Mumbai/Pune conversational style.

Input Question: {question}
Input Answer: {answer}

Provide your output strictly in JSON format as follows:
{{
  "code_mixed_question": "...",
  "code_mixed_answer": "..."
}}

Output ONLY the JSON, no other text."""

def save_dataset(output_file, dataset):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def generate_synthetic_data(input_file, output_file, limit=5000):
    api_key = load_api_key()
    client = Groq(api_key=api_key)
    current_key_suffix = api_key[-6:]
    print(f"Loaded Groq API key: ...{current_key_suffix}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Resume support
    generated_dataset = []
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                generated_dataset = json.load(f)
                existing_ids = {item["original_id"] for item in generated_dataset}
            print(f"Resuming from {len(generated_dataset)} saved samples.")
        except Exception as e:
            print(f"Could not load existing output ({e}). Starting fresh.")
            generated_dataset = []

    remaining = limit - len(generated_dataset)
    print(f"Target: {limit} samples | Already done: {len(generated_dataset)} | Remaining: {remaining}")
    print(f"Model: {MODEL_NAME} | Delay: {REQUEST_DELAY}s/request")
    print(f"Estimated time: ~{remaining * (REQUEST_DELAY + 1) / 60:.0f} minutes\n")

    pbar = tqdm(total=min(limit, len(raw_data)), desc="Generating Code-Mixed Data", initial=len(generated_dataset))

    consecutive_errors = 0
    save_every = 5

    for item in raw_data:
        if len(generated_dataset) >= limit:
            print(f"\nReached target of {limit} samples!")
            break

        if consecutive_errors >= 15:
            tqdm.write(f"\n[STOPPED] 15 consecutive errors. Saving progress and stopping.")
            break

        if item["id"] in existing_ids:
            continue

        prompt = PROMPT_TEMPLATE.format(
            question=item["question"],
            answer=item["answer"]
        )

        success = False
        max_retries = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You output valid JSON only. No markdown, no code blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    response_format={"type": "json_object"}
                )

                raw_text = response.choices[0].message.content.strip()

                # Clean markdown wrapping if present
                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                if raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                res_json = json.loads(raw_text)

                generated_dataset.append({
                    "original_id": item["id"],
                    "original_question": item["question"],
                    "original_answer": item["answer"],
                    "code_mixed_question": res_json.get("code_mixed_question", ""),
                    "code_mixed_answer": res_json.get("code_mixed_answer", ""),
                    "source": item.get("source", "PubMedQA")
                })
                existing_ids.add(item["id"])
                consecutive_errors = 0
                success = True
                pbar.update(1)

                # Periodic save
                if len(generated_dataset) % save_every == 0:
                    save_dataset(output_file, generated_dataset)

                time.sleep(REQUEST_DELAY)
                break  # Success

            except Exception as e:
                err_msg = str(e)

                if "429" in err_msg or "rate_limit" in err_msg.lower() or "RESOURCE_EXHAUSTED" in err_msg:
                    # Rate limited — wait 60s (Groq resets per-minute quotas every 60s)
                    wait = 65
                    tqdm.write(f"[RATE LIMIT] Attempt {attempt+1}/{max_retries}. Waiting {wait}s for quota reset...")
                    time.sleep(wait)

                    # Re-read key from file in case user swapped it during the wait
                    try:
                        new_key = load_api_key()
                        if new_key[-6:] != current_key_suffix:
                            client = Groq(api_key=new_key)
                            current_key_suffix = new_key[-6:]
                            tqdm.write(f"[KEY SWAPPED] Now using key: ...{current_key_suffix}")
                    except Exception:
                        pass  # Keep using current key

                elif "401" in err_msg or "invalid_api_key" in err_msg.lower():
                    tqdm.write(f"[AUTH ERROR] Key ...{current_key_suffix} is invalid.")
                    tqdm.write(f"[AUTH ERROR] Update keys.txt with a valid key. Waiting 30s for you to swap...")
                    time.sleep(30)
                    # Try to reload key
                    try:
                        new_key = load_api_key()
                        if new_key[-6:] != current_key_suffix:
                            client = Groq(api_key=new_key)
                            current_key_suffix = new_key[-6:]
                            tqdm.write(f"[KEY SWAPPED] Now using key: ...{current_key_suffix}")
                        else:
                            tqdm.write(f"[AUTH ERROR] Same invalid key still in file. Stopping.")
                            consecutive_errors = 15  # Force stop
                            break
                    except Exception:
                        consecutive_errors = 15
                        break

                else:
                    tqdm.write(f"[ERROR] Item {item['id']}: {err_msg[:150]}")
                    consecutive_errors += 1
                    time.sleep(5)
                    break  # Don't retry parse/other errors

        if not success:
            consecutive_errors += 1

    pbar.close()
    save_dataset(output_file, generated_dataset)
    print(f"\nDone! {len(generated_dataset)} total samples saved to {output_file}")

if __name__ == "__main__":
    raw_path = os.path.join(ROOT, "data", "raw", "pubmed_qa_raw.json")
    out_path = os.path.join(ROOT, "data", "processed", "synthetic_codemix_qa.json")
    # Generating 5000 samples for Phase 2 corpus requirements
    generate_synthetic_data(raw_path, out_path, limit=5000)