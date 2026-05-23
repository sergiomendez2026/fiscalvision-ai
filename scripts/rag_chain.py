"""
rag_chain.py — Cadena RAG para consultas tributarias del SRI Ecuador

Usa:
  - GPT-4o (OpenAI) como LLM generador
  - text-embedding-3-small para búsqueda semántica
  - ChromaDB como vector store local
  - Memoria de conversación de ventana deslizante (últimas 5 interacciones)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Constantes ────────────────────────────────────────────────────────────────

CHROMA_DIR    = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))
OPENAI_MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")
TOP_K_CHUNKS  = 5   # cuántos chunks recuperar por consulta

# ── Prompt especializado ──────────────────────────────────────────────────────

SYSTEM_PROMPT = PromptTemplate.from_template("""
Eres un asistente experto en tributación ecuatoriana, especializado en los \
documentos oficiales del Servicio de Rentas Internas (SRI) del Ecuador.

INSTRUCCIONES:
1. Responde ÚNICAMENTE basándote en el contexto proporcionado de los documentos del SRI.
2. Cita siempre la fuente: artículo, capítulo o nombre del documento donde encontraste la información.
3. Si la pregunta no puede responderse con los documentos disponibles, dilo claramente: 
   "Esta información no se encuentra en los documentos cargados. Te recomiendo consultar directamente en sri.gob.ec"
4. Usa un lenguaje claro y accesible, evitando jerga innecesaria.
5. Si hay cambios recientes (especialmente relacionados con comprobantes electrónicos 2025), prioriza esa información.
6. Nunca inventes cifras, porcentajes o fechas. Si no estás seguro, dilo.

CONTEXTO DE DOCUMENTOS SRI:
{context}

HISTORIAL DE CONVERSACIÓN:
{chat_history}

PREGUNTA DEL USUARIO:
{question}

RESPUESTA (con fuente citada):""")

# ── Builder ───────────────────────────────────────────────────────────────────

def build_chain():
    """
    Construye y retorna la cadena RAG completa.
    Llamar una vez al iniciar la app (carga el vector store en memoria).
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el vector store en '{CHROMA_DIR}'. "
            "Ejecuta primero: python scripts/ingest.py"
        )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_CHUNKS},
    )

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,          # máxima consistencia para consultas legales
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Ventana de 5 turnos: recuerda el contexto de la conversación actual
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,   # para mostrar fuentes en la UI
        combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
    )

    return chain


def format_sources(source_docs) -> str:
    """Formatea los documentos fuente para mostrarlos en la UI."""
    if not source_docs:
        return ""

    seen = set()
    lines = []
    for doc in source_docs:
        meta = doc.metadata
        label = meta.get("source_type", "Documento SRI")
        fname = meta.get("file_name", "")
        page  = meta.get("page", "")
        key   = f"{label}|{fname}|{page}"
        if key not in seen:
            seen.add(key)
            page_str = f" — pág. {page + 1}" if page != "" else ""
            lines.append(f"• **{label}** ({fname}{page_str})")

    return "\n".join(lines)
