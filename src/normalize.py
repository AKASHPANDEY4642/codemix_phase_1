# src/normalize.py
import unicodedata

def normalize_text(text: str) -> str:
    """
    Applies NFC Unicode normalization and strips extra spaces.
    Preserves Devanagari script integrity.
    """
    if not text:
        return ""
    
    # Apply standard Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)
    
    # Remove redundant whitespace
    normalized = " ".join(normalized.split())
    
    return normalized

if __name__ == "__main__":
    sample_text = "मला  stomach pain   होतोय "
    cleaned = normalize_text(sample_text)
    print("Original:", repr(sample_text))
    print("Cleaned: ", repr(cleaned))