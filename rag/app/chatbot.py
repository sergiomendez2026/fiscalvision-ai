import streamlit as st

st.set_page_config(
    page_title="FiscalVision RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ FiscalVision RAG Assistant")

st.markdown("""
### Plataforma de inteligencia tributaria impulsada por IA

Este asistente utiliza:
- Retrieval Augmented Generation (RAG)
- Inteligencia Artificial
- Machine Learning
- Documentos tributarios verificables
- Búsqueda semántica
""")

question = st.text_input(
    "Escribe tu pregunta tributaria:"
)

if question:
    st.success(f"Pregunta recibida: {question}")

    st.info(
        "Próximo paso: conectar embeddings, vector database y documentos oficiales."
    )
