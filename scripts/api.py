"""
api.py — API REST con FastAPI para integrar el chatbot en FiscalVision AI

Endpoints:
  POST /chat          → pregunta + session_id → respuesta + fuentes
  DELETE /chat/{sid}  → limpia la memoria de una sesión
  GET  /health        → estado del servicio

Ejecutar:
  uvicorn scripts.api:app --reload --port 8001
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))
from rag_chain import build_chain, format_sources

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FiscalVision AI — Chatbot Tributario",
    description="API RAG sobre documentos del SRI Ecuador",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restringe en producción a tu dominio
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caché de cadenas por session_id (en producción usar Redis)
_chains: dict = {}

# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: str
    session_id: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    chroma_ok = Path(os.getenv("CHROMA_DIR", "data/chroma_db")).exists()
    return {
        "status":    "ok" if chroma_ok else "degraded",
        "chroma_db": chroma_ok,
        "model":     os.getenv("OPENAI_MODEL", "gpt-4o"),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    # Obtiene o crea la cadena RAG para esta sesión
    if req.session_id not in _chains:
        try:
            _chains[req.session_id] = build_chain()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    chain = _chains[req.session_id]

    result  = chain({"question": req.question})
    answer  = result["answer"]
    sources = format_sources(result.get("source_documents", []))

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=req.session_id,
    )


@app.delete("/chat/{session_id}")
def clear_session(session_id: str):
    if session_id in _chains:
        del _chains[session_id]
    return {"message": f"Sesión '{session_id}' eliminada."}
