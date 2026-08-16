# src/test_full_retrieval.py
from retriever import HybridRetriever

def test_full_kb():
    retriever = HybridRetriever()
    
    test_queries = [
        "metformin che side effects kay ahet",
        "diabetes sathi best treatment kay ahe",
        "mala stomach pain ani fever hotoy"
    ]
    
    print("=" * 80)
    print("PHASE 5: FULL KNOWLEDGE BASE HYBRID RETRIEVAL TEST")
    print("=" * 80)
    
    for idx, q in enumerate(test_queries, 1):
        print(f"\nQUERY #{idx}: '{q}'")
        res = retriever.retrieve(q, top_k=3)
        
        print(f"Extracted Medical Terms: {res['query_info']['medical_entities']}")
        print(f"Normalized Query:        {res['query_info']['normalized_query']}")
        print("\nTop 3 Retrieved Contexts from Full KB:")
        
        for rank, ctx in enumerate(res['retrieved_context'], 1):
            print(f"\n  [{rank}] Source ID: {ctx['doc_id']} ({ctx['source']}) | RRF Score: {ctx['rrf_score']:.4f}")
            print(f"      Text: {ctx['text'][:200]}...")
            
        print("-" * 80)

if __name__ == "__main__":
    test_full_kb()