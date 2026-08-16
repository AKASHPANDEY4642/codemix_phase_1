# Import the Gradio library for building the web UI.
import gradio as gr

# Import the hybrid medical retriever.
from retriever import HybridRetriever

# Import the medical LLM generator.
from generator import MedicalGenerator

# Import the risk assessment and safety guardrail system.
from risk_scorer import RiskAssessor

# Import SQLite database initialization and query logging functions.
from db_logger import init_db, log_query


# Display startup message.
print("Initializing Models (This takes a few seconds)...")

# Initialize the medical knowledge-base retriever.
retriever = HybridRetriever()

# Initialize the Groq-based medical answer generator.
generator = MedicalGenerator()

# Initialize the medical risk assessor.
risk_assessor = RiskAssessor()


# Initialize the SQLite database.
init_db()

# Display that the complete system is ready.
print("System Ready!")


def process_query(user_message, history):
    """
    Process a user query through the complete RAG pipeline.
    """

    # 1. Assess the medical risk level of the user query.
    risk_level = risk_assessor.assess_risk(user_message)

    # 2. Retrieve relevant medical knowledge from the knowledge base.
    retrieval_res = retriever.retrieve(user_message, top_k=3)

    # Extract the retrieved contexts.
    contexts = retrieval_res["retrieved_context"]

    # 3. Generate an answer using the retrieved medical contexts.
    raw_answer = generator.generate_answer(user_message, contexts)

    # 4. Apply safety guardrails based on the detected risk level.
    safe_answer = risk_assessor.apply_guardrails(
        raw_answer,
        risk_level
    )

    # 5. Create the citation section for the UI.
    citations = "\n\n### 📚 संदर्भ (Citations):\n"

    # Process every retrieved context.
    for i, ctx in enumerate(contexts, 1):

        # Extract a short preview of the retrieved text.
        snippet = (
            ctx["text"][:150]
            .replace("\n", " ")
            + "..."
        )

        # Add the source and document ID to the citation section.
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

    # 6. Save the interaction in the SQLite database.
    log_query(
        user_message,
        risk_level,
        final_response
    )

    # Return the final answer to Gradio.
    return final_response


# Create the Gradio application.
with gr.Blocks() as demo:

    # Display the application title.
    gr.Markdown("# 🩺 MedManglish-RAG")

    # Display the application description.
    gr.Markdown(
        "A Script-Aware Retrieval-Augmented Generation Framework "
        "for Marathi-English Code-Mixed Medical QA"
    )

    # Create the chat interface.
    chat_interface = gr.ChatInterface(
        fn=process_query,

        # Configure the chatbot height.
        chatbot=gr.Chatbot(height=500),

        # Configure the user input box.
        textbox=gr.Textbox(
            placeholder=(
                "Ask a medical question in Manglish "
                "(e.g., 'mala stomach pain hotoy')..."
            ),
            container=False,
            scale=7,
        ),

        # Configure the chat title.
        title="Ask the Medical AI",

        # Configure the chat description.
        description=(
            "This system understands Marathi-English "
            "code-mixed queries. "
            "**Not a substitute for professional medical advice.**"
        ),

        # Provide example questions.
        examples=[
            "vitamins kashasathi ghyave?",
            "metformin che side effects kay ahet",
            "mala chhatit khup pain hotoy, heart attack ahe ka?",
        ],
    )


# Run the application only when this file is executed directly.
if __name__ == "__main__":

    # Launch Gradio and apply the theme here.
    # Gradio 6 requires theme to be passed to launch().
    demo.launch(
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
