# src/transliterate.py
import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from medical_terms import DEFAULT_MEDICAL_TERMS

def is_devanagari(char: str) -> bool:
    """Checks if character is in Devanagari Unicode range."""
    return '\u0900' <= char <= '\u097F'

def detect_script_type(text: str) -> str:
    """Classifies text as 'devanagari', 'roman', or 'mixed'."""
    has_dev = any(is_devanagari(c) for c in text)
    has_lat = any('a' <= c.lower() <= 'z' for c in text)
    
    if has_dev and has_lat:
        return "mixed"
    elif has_dev:
        return "devanagari"
    else:
        return "roman"

def transliterate_roman_to_devanagari(text: str, medical_vocab: set = DEFAULT_MEDICAL_TERMS) -> str:
    """
    Transliterates Roman Marathi words to Devanagari while preserving 
    English medical terms in English.
    """
    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        # Strip punctuation for check
        clean_token = re.sub(r'[^\w\s]', '', token).lower()
        
        # If the word is already Devanagari or is a preserved English medical term
        if any(is_devanagari(c) for c in token) or clean_token in medical_vocab:
            processed_tokens.append(token)
        else:
            # Transliterate Roman Marathi to Devanagari (ITRANS scheme)
            try:
                dev_token = transliterate(token.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
                processed_tokens.append(dev_token)
            except Exception:
                processed_tokens.append(token)
                
    return " ".join(processed_tokens)

if __name__ == "__main__":
    test_queries = [
        "majhya potaat dukhte",
        "mala stomach pain hotoy",
        "BP high ahe tar kay karave"
    ]
    
    for q in test_queries:
        print(f"Original:      {q}")
        print(f"Script Type:   {detect_script_type(q)}")
        print(f"Transliterated:{transliterate_roman_to_devanagari(q)}\n")