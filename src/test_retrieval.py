# src/test_retrieval.py
from retriever import HybridRetriever

def run_retrieval_test():
    retriever = HybridRetriever()
    
    test_queries = [
        "metformin che side effects kay ahet",
        "mala stomach pain hotoy",
    ]
    
    for idx, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"QUERY #{idx}: '{query}'")
        print("=" * 80)
        
        results = retriever.retrieve(query)
        
        print(f"\n1. Extracted Medical Terms: {results['query_info']['medical_entities']}")
        print(f"2. Normalized Query:        {results['query_info']['normalized_query']}")
        print(f"3. HyDE Document Snippet:   {results['hyde_document'][:100]}...")
        
        print("\n4. Top Retrieved Contexts (After RRF):")
        for i, context in enumerate(results['retrieved_context'], 1):
            print(f"   [{i}] {context}")
        print("\n")

if __name__ == "__main__":
    run_retrieval_test()