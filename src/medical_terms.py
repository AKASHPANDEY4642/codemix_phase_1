# src/medical_terms.py
import re

# Comprehensive list of medical terms, symptoms, and drugs to preserve in English
DEFAULT_MEDICAL_TERMS = {
    "fever", "headache", "stomach", "pain", "cough", "cold", "diabetes", 
    "blood", "pressure", "bp", "sugar", "metformin", "paracetamol", "cancer",
    "infection", "virus", "bacterial", "ct", "scan", "mri", "xray", "x-ray",
    "heart", "attack", "stroke", "cholesterol", "thyroid", "asthma", "allergy",
    "tablet", "capsule", "syrup", "injection", "dose", "doctor", "hospital",
    "symptom", "side", "effects", "treatment", "medicine", "vitamin", "multivitamin",
    "pulse", "oxygen", "icu", "opd", "surgery", "operation", "tumor", "ulcer"
}

def extract_english_medical_terms(text: str, custom_dict: set = None) -> set:
    """Extracts medical terms present in the text that must remain in English."""
    medical_vocab = custom_dict if custom_dict else DEFAULT_MEDICAL_TERMS
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    found_terms = {word for word in words if word in medical_vocab}
    return found_terms

if __name__ == "__main__":
    sample = "Mala stomach pain ahe ani BP high ahe"
    found = extract_english_medical_terms(sample)
    print("Sample:", sample)
    print("Extracted English Medical Terms:", found)