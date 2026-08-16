# src/script_normalization.py
import re
import unicodedata
from transliterate import detect_script_type, transliterate_roman_to_devanagari
from medical_terms import extract_english_medical_terms, DEFAULT_MEDICAL_TERMS

def normalize_orthography(text: str) -> str:
    """Normalizes Unicode variations, whitespace, and Devanagari conjuncts."""
    # Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_query_pipeline(query: str) -> dict:
    """
    Complete Phase 3 Pipeline:
    1. Detect Script Type
    2. Extract Medical Entities
    3. Normalize Orthography
    4. Transliterate Roman Marathi -> Devanagari (preserving English medical terms)
    """
    query_clean = normalize_orthography(query)
    script_type = detect_script_type(query_clean)
    medical_entities = extract_english_medical_terms(query_clean)
    
    # Transliterate Roman Marathi parts if script is Roman or Mixed
    if script_type in ["roman", "mixed"]:
        normalized_query = transliterate_roman_to_devanagari(query_clean)
    else:
        normalized_query = query_clean
        
    return {
        "raw_query": query,
        "script_type": script_type,
        "medical_entities": list(medical_entities),
        "normalized_query": normalized_query
    }

if __name__ == "__main__":
    sample_query = "mla stomach pain hotoy ani fever pan ahe"
    result = process_query_pipeline(sample_query)
    
    print("=== PHASE 3 NORMALIZATION PIPELINE OUTPUT ===")
    for key, value in result.items():
        print(f"{key:<20}: {value}")