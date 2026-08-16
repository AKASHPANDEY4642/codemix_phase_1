import gradio as gr

from retriever import HybridRetriever
from generator import MedicalGenerator
from risk_scorer import RiskAssessor


# Initialize all required models and components.
print("Initializing Models (This takes a few seconds)...")

retriever = HybridRetriever()
generator = MedicalGenerator()
risk_assessor = RiskAssessor()

print("System Ready!")


def process_query(user_message, history):
    """
    Process one user medical query through the complete RAG pipeline.
    """

    # 1. Assess the medical risk level of the user query.
    risk_level = risk_assessor.assess_risk(user_message)

    # 2. Retrieve the most relevant medical knowledge.
    retrieval_res = retriever.retrieve(user_message, top_k=3)

    # Extract retrieved context documents.
    contexts = retrieval_res["retrieved_context"]

    # 3. Generate an answer using the retrieved medical context.
    raw_answer = generator.generate_answer(user_message, contexts)

    # 4. Apply safety guardrails based on the detected risk level.
    safe_answer = risk_assessor.apply_guardrails(raw_answer, risk_level)

    # 5. Create citation text for the retrieved documents.
    citations = "\n\n### 📚 संदर्भ (Citations):\n"

    # Add each retrieved document as a citation.
    for i, ctx in enumerate(contexts, 1):

        # Take only the first 150 characters for the UI preview.
        snippet = ctx["text"][:150].replace("\n", " ") + "..."

        # Add source, document ID, and text snippet.
        citations += (
            f'**[{i}]** {ctx["source"]} '
            f'(ID: {ctx["doc_id"]})\n'
            f'*"{snippet}"*\n\n'
        )

    # Combine risk level, answer, and citations.
    final_response = (
        f"**⚠️ Risk Level:** {risk_level}\n\n"
        f"{safe_answer}"
        f"{citations}"
    )

    # Return the final response to Gradio.
    return final_response


# Build the Gradio user interface.
with gr.Blocks() as demo:

    # Application title.
    gr.Markdown("# 🩺 MedManglish-RAG")

    # Application description.
    gr.Markdown(
        "A Script-Aware Retrieval-Augmented Generation Framework "
        "for Marathi-English Code-Mixed Medical QA"
    )

    # Chat interface.
    chat_interface = gr.ChatInterface(
        fn=process_query,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(
            placeholder=(
                "Ask a medical question in Manglish "
                "(e.g., 'mala stomach pain hotoy')..."
            ),
            container=False,
            scale=7,
        ),
        title="Ask the Medical AI",
        description=(
            "This system understands Marathi-English code-mixed queries. "
            "**Not a substitute for professional medical advice.**"
        ),
        examples=[
            "vitamins kashasathi ghyave?",
            "metformin che side effects kay ahet",
            "mala chhatit khup pain hotoy, heart attack ahe ka?",
        ],
    )


# Start the application.
if __name__ == "__main__":
    demo.launch(
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )