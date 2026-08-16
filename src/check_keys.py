import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS_FILE = ROOT / "keys.txt"

if not KEYS_FILE.exists():
    print("keys.txt not found at", KEYS_FILE)
    raise SystemExit(1)

keys = [k.strip() for k in KEYS_FILE.read_text(encoding='utf-8').splitlines() if k.strip()]
print(f"Found {len(keys)} keys (showing first 3):")
for i, k in enumerate(keys[:3], start=1):
    print(f"{i}: {k}")

try:
    from google import genai
except Exception as e:
    print("google-genai not installed or import failed:", e)
    print("Install with: python -m pip install -r requirements.txt")
    raise SystemExit(0)

print("google-genai is importable. Testing first key...")
from google.genai import Client

for i, key in enumerate(keys[:5], start=1):
    try:
        client = Client(api_key=key)
        resp = client.models.generate_content(model="gemini-2.5-flash", contents="Reply: OK")
        print(f"Key {i}: OK, response length {len(getattr(resp, 'text', str(resp))) }")
    except Exception as e:
        print(f"Key {i}: FAIL -> {e}")
