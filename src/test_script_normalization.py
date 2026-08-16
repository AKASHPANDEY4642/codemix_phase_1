# src/test_script_normalization.py
from script_normalization import process_query_pipeline

def run_phase3_evaluation():
    test_cases = [
        {"type": "Roman Marathi", "query": "majhya potaat dukhte"},
        {"type": "Devanagari Marathi", "query": "माझ्या पोटात दुखते"},
        {"type": "Code-Mixed Roman", "query": "mala stomach pain hotoy"},
        {"type": "Code-Mixed Devanagari", "query": "मला stomach pain होतोय"},
        {"type": "Medical/Drug Query", "query": "metformin che side effects kay ahet"}
    ]
    
    print("=" * 80)
    print("PHASE 3: SCRIPT NORMALIZATION & TRANSLITERATION TEST SUITE")
    print("=" * 80)
    
    for idx, tc in enumerate(test_cases, 1):
        res = process_query_pipeline(tc["query"])
        print(f"Test #{idx} [{tc['type']}]")
        print(f"  Input Query:        {res['raw_query']}")
        print(f"  Detected Script:    {res['script_type']}")
        print(f"  Extracted Medical:  {res['medical_entities']}")
        print(f"  Normalized Output:  {res['normalized_query']}")
        print("-" * 80)

if __name__ == "__main__":
    run_phase3_evaluation()