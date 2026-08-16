# src/safe_rag.py
from retriever import HybridRetriever
from generator import MedicalGenerator
from risk_scorer import RiskAssessor

def run_safe_rag():
    print("Loading Safe MedManglish-RAG Pipeline...")
    retriever = HybridRetriever()
    generator = MedicalGenerator()
    risk_assessor = RiskAssessor()
    
    test_queries = [
        "vitamins kashasathi ghyave?",                        # Expected: LOW Risk
        "metformin che side effects kay ahet",                # Expected: MEDIUM Risk
        "mala chhatit (chest) khup pain hotoy, heart attack ahe ka?" # Expected: HIGH Risk
    ]
    
    print("\n" + "=" * 80)
    print("PHASE 7: SAFE RAG PIPELINE TEST")
    print("=" * 80)
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[USER QUERY #{idx}]: '{query}'")
        print("-" * 80)
        
        # 1. Risk Assessment
        risk_level = risk_assessor.assess_risk(query)
        print(f"➜ Detected Risk Level: {risk_level}")
        
        # 2. Retrieval
        retrieval_res = retriever.retrieve(query, top_k=3)
        contexts = retrieval_res["retrieved_context"]
        
        # 3. Generation
        raw_answer = generator.generate_answer(query, contexts)
        
        # 4. Apply Safety Guardrails
        safe_answer = risk_assessor.apply_guardrails(raw_answer, risk_level)
        
        print("\n>>> FINAL SAFE ANSWER <<<")
        print(safe_answer)
        print("=" * 80)

if __name__ == "__main__":
    run_safe_rag()