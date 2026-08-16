# src/generator.py
import os
from groq import Groq

class MedicalGenerator:
    def __init__(self):
        print("Initializing LLM Generator via Groq...")
        with open("keys.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def generate_answer(self, raw_query: str, retrieved_contexts: list) -> str:
        # 1. Format the retrieved context with clear citation numbers
        context_text = ""
        for i, ctx in enumerate(retrieved_contexts, 1):
            context_text += f"[{i}] (Source: {ctx['source']}, ID: {ctx['doc_id']})\n{ctx['text']}\n\n"
        
        # 2. Build the strict Generation Prompt
        prompt = f"""You are a helpful medical AI assistant. Answer the user's question using ONLY the provided contexts.
        
        RULES:
        1. Language: Write the answer in natural code-mixed Marathi-English (Manglish).
        2. Vocabulary: Keep all medical terms (e.g., diseases, drugs, body parts) strictly in English. Write the conversational grammar and connecting words in Marathi (Devanagari script).
        3. Citations: You MUST cite the source of your information using [1], [2], etc., exactly matching the provided context numbers. Place the citation at the end of the relevant sentence.
        4. Honesty: If the context does not contain enough information to answer the question, do not guess. Say exactly: "मला याबद्दल पुरेशी माहिती नाही. कृपया doctor ला भेटा."
        
        CONTEXTS:
        {context_text}
        
        USER QUESTION:
        {raw_query}
        
        ANSWER:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate, bilingual medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3, # Low temperature for factual RAG tasks
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error] LLM Generation failed: {e}"