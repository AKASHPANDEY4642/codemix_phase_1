import time
from pathlib import Path
from google import genai

ROOT = Path(__file__).resolve().parents[1]
KEYS_FILE = ROOT / "keys.txt"

with open(KEYS_FILE, "r", encoding="utf-8") as f:
    keys = [k.strip() for k in f.readlines() if k.strip()]

print(f"Testing {len(keys)} API keys...")

working_keys = []
quota_keys = []
denied_keys = []
other_keys = []

for i, key in enumerate(keys, 1):
    masked = f"...{key[-6:]}"
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents="Reply: OK")
        print(f"Key {i} ({masked}): SUCCESS -> {resp.text.strip()}")
        working_keys.append(key)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            print(f"Key {i} ({masked}): 429 Rate Limit / Quota Exceeded")
            quota_keys.append(key)
        elif "403" in err or "PERMISSION_DENIED" in err:
            print(f"Key {i} ({masked}): 403 Permission Denied")
            denied_keys.append(key)
        else:
            print(f"Key {i} ({masked}): Error -> {err[:80]}")
            other_keys.append(key)

print("\n--- Summary ---")
print(f"Working keys: {len(working_keys)}")
print(f"Quota/Rate limited (429): {len(quota_keys)}")
print(f"Permission denied (403): {len(denied_keys)}")
print(f"Other errors: {len(other_keys)}")
