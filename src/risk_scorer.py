# src/risk_scorer.py
import os
from groq import Groq

class RiskAssessor:
    def __init__(self):
        print("Initializing Risk Assessor...")
        with open("keys.txt", "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"

    def assess_risk(self, query: str) -> str:
        """Classifies the medical query into LOW, MEDIUM, or HIGH risk."""
        prompt = f"""Analyze this medical query and classify its risk level.
        
        RULES:
        LOW: General health info, diet, vitamins, lifestyle.
        MEDIUM: Common symptoms, medication side effects, general treatments.
        HIGH: Emergencies, severe pain, drug interactions, life-threatening conditions, surgery.

        Query: "{query}"

        Respond ONLY with ONE word: LOW, MEDIUM, or HIGH."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Zero temperature for deterministic classification
                max_tokens=10
            )
            risk = response.choices[0].message.content.strip().upper()
            if "HIGH" in risk: return "HIGH"
            if "MEDIUM" in risk: return "MEDIUM"
            return "LOW"
        except Exception as e:
            print(f"[Risk Error] {e}")
            return "HIGH" # Default to HIGH risk if classification fails (Safety First)

    def apply_guardrails(self, answer: str, risk_level: str) -> str:
        """Appends appropriate Marathi medical disclaimers based on risk level."""
        disclaimer = ""
        if risk_level == "HIGH":
            disclaimer = "\n\n⚠️ गंभीर चेतावणी: ही माहिती केवळ संदर्भासाठी आहे. कृपया तात्काळ doctor ला भेटा (Consult a doctor immediately)."
        elif risk_level == "MEDIUM":
            disclaimer = "\n\n📋 टीप: हा सामान्य सल्ला आहे. कोणतेही औषध घेण्यापूर्वी doctor चा सल्ला घ्या (Consult a doctor before taking medication)."
        else:
            disclaimer = "\n\n💡 माहिती: हा केवळ सामान्य आरोग्य सल्ला आहे."
            
        return answer + disclaimer