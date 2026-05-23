"""
app.py — Interfaz Streamlit del Chatbot Tributario SRI Ecuador
Integrado con FiscalVision AI

Ejecutar:
  streamlit run scripts/app.py
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Asegura que el módulo rag_chain sea importable desde cualquier directorio
sys.path.insert(0, str(Path(__file__).parent))
from rag_chain import build_chain, format_sources

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Visión Fiscal AI — Consultas SRI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos personalizados ────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Burbuja de fuentes */
    .source-box {
        background-color: #f0f4f8;
        border-left: 3px solid #1a56db;
        border-radius: 4px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #374151;
        margin-top: 6px;
    }
    /* Aviso de disclaimer */
    .disclaimer {
        background-color: #fffbeb;
        border-left: 3px solid #f59e0b;
        border-radius: 4px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://www.sri.gob.ec/o/sri-portlet-biblioteca-alfresco-internet/images/logo_sri.png",
             width=140)
    st.markdown("## 🧾 Visión Fiscal AI")
    st.markdown("Consultas sobre legislación tributaria ecuatoriana basadas en documentos oficiales del SRI.")

    st.divider()

    st.markdown("### 📚 Base de conocimiento")
    docs_info = {
        "Ley de Régimen Tributario Interno": "LRTI vigente",
        "Reglamento a la LRTI": "Decreto Ejecutivo",
        "Guías y Circulares SRI": "Resoluciones NAC",
        "Comprobantes Electrónicos": "Ficha técnica mar-2025",
    }
    for doc, desc in docs_info.items():
        st.markdown(f"✅ **{doc}**  \n<small>{desc}</small>", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain    = None
        st.rerun()

    st.divider()

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Aviso legal</b><br>
    Las respuestas se generan a partir de documentos del SRI. 
    Para decisiones fiscales importantes, consulta a un profesional tributario certificado.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Parte de [FiscalVision AI](https://github.com/sergiomendez2026/fiscalvision-ai)")

# ── Validaciones ──────────────────────────────────────────────────────────────

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ **OPENAI_API_KEY** no encontrada. Crea un archivo `.env` con tu clave.")
    st.code("OPENAI_API_KEY=sk-proj-...", language="bash")
    st.stop()

chroma_dir = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))
if not chroma_dir.exists():
    st.error("❌ **Vector store no encontrado.** Ejecuta primero el script de ingestión:")
    st.code("python scripts/ingest.py", language="bash")
    st.stop()

# ── Estado de sesión ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# ── Carga del modelo ──────────────────────────────────────────────────────────

if st.session_state.chain is None:
    with st.spinner("⚡ Cargando base de conocimiento tributaria..."):
        try:
            st.session_state.chain = build_chain()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

# ── Encabezado principal ──────────────────────────────────────────────────────

st.title("🧾 Chatbot Tributario — SRI Ecuador")
st.caption("Respuestas basadas en la LRTI, Reglamento, Circulares y Ficha de Comprobantes Electrónicos (mar-2025)")

# ── Preguntas de ejemplo ──────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("### 💡 Preguntas frecuentes")
    example_questions = [
        "¿Cuál es la tarifa del IVA para servicios digitales en Ecuador?",
        "¿Qué documentos son obligatorios en una factura electrónica según la ficha técnica 2025?",
        "¿Cómo se calcula el Impuesto a la Renta para personas naturales?",
        "¿Cuáles son los plazos para declarar el IVA mensual?",
        "¿Qué es el RIMPE y quiénes aplican a este régimen?",
        "¿Cuáles son las claves de acceso en los comprobantes electrónicos?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

# ── Historial de mensajes ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Fuentes consultadas", expanded=False):
                st.markdown(
                    f'<div class="source-box">{msg["sources"]}</div>',
                    unsafe_allow_html=True,
                )

# ── Manejar pregunta de ejemplo pendiente ─────────────────────────────────────

pending = st.session_state.pop("pending_question", None)

# ── Input del usuario ─────────────────────────────────────────────────────────

user_input = st.chat_input("Escribe tu consulta tributaria aquí...")
question   = pending or user_input

if question:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consultando documentos del SRI..."):
            result  = st.session_state.chain({"question": question})
            answer  = result["answer"]
            sources = format_sources(result.get("source_documents", []))

        st.markdown(answer)

        if sources:
            with st.expander("📄 Fuentes consultadas", expanded=False):
                st.markdown(
                    f'<div class="source-box">{sources}</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
