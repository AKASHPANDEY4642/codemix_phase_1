# src/test_rag.py
from retriever import HybridRetriever
from generator import MedicalGenerator

def run_end_to_end_rag():
    print("Loading End-to-End RAG Pipeline...")
    retriever = HybridRetriever()
    generator = MedicalGenerator()
    
    test_queries = [
        "metformin che side effects kay ahet",
        "diabetes sathi best treatment kay ahe"
    ]
    
    print("\n" + "=" * 80)
    print("PHASE 6: END-TO-END RAG GENERATION TEST")
    print("=" * 80)
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[USER QUERY #{idx}]: '{query}'")
        print("-" * 80)
        
        # Step A: Retrieve Context (Phase 5)
        print("1. Retrieving context...")
        retrieval_res = retriever.retrieve(query, top_k=3)
        contexts = retrieval_res["retrieved_context"]
        
        # Step B: Generate Answer (Phase 6)
        print("2. Generating code-mixed answer with citations...")
        final_answer = generator.generate_answer(query, contexts)
        
        # Display Results
        print("\n>>> FINAL ANSWER <<<")
        print(final_answer)
        
        print("\n>>> CITED SOURCES <<<")
        for i, ctx in enumerate(contexts, 1):
            print(f"[{i}] ID: {ctx['doc_id']} | Text: {ctx['text'][:120]}...")
            
        print("=" * 80)

if __name__ == "__main__":
    run_end_to_end_rag()