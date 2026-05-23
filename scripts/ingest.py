"""
ingest.py — Pipeline de ingestión para documentos tributarios del SRI Ecuador

Documentos soportados:
  - Ley de Régimen Tributario Interno (LRTI)        → PDF
  - Reglamento a la LRTI                            → PDF
  - Guías y circulares del SRI                      → PDF
  - Ficha Técnica de Comprobantes Electrónicos      → PDF (marzo 2025)

Uso:
  python scripts/ingest.py
  python scripts/ingest.py --reset   # limpia y re-indexa todo
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────

DOCS_DIR  = Path(os.getenv("DOCS_DIR",  "data/docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "data/chroma_db"))

# Subcarpetas esperadas dentro de data/docs/
# Crea estas carpetas y pon tus PDFs ahí
SUBFOLDERS = {
    "lrti":               "Ley de Régimen Tributario Interno",
    "reglamento":         "Reglamento a la LRTI",
    "guias_circulares":   "Guías y Circulares del SRI",
    "comprobantes":       "Ficha Técnica Comprobantes Electrónicos",
}

# Tamaño de chunk optimizado para artículos de ley:
# 600 tokens captura un artículo completo sin cortarlo
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── Funciones ─────────────────────────────────────────────────────────────────

def load_documents():
    """Carga todos los PDFs organizados por subcarpeta y les asigna metadatos de fuente."""
    all_docs = []

    for folder_name, source_label in SUBFOLDERS.items():
        folder_path = DOCS_DIR / folder_name
        if not folder_path.exists():
            print(f"  ⚠️  Carpeta no encontrada (se omite): {folder_path}")
            continue

        pdf_files = list(folder_path.glob("**/*.pdf"))
        if not pdf_files:
            print(f"  ⚠️  Sin PDFs en: {folder_path}")
            continue

        for pdf_path in pdf_files:
            print(f"  📄 Cargando: {pdf_path.name}")
            loader = PyMuPDFLoader(str(pdf_path))
            docs = loader.load()

            # Enriquecer metadatos para citar la fuente en las respuestas
            for doc in docs:
                doc.metadata["source_type"] = source_label
                doc.metadata["file_name"]   = pdf_path.name

            all_docs.extend(docs)

    return all_docs


def chunk_documents(docs):
    """
    Divide los documentos en chunks.
    Usa separadores pensados para leyes: doble salto de línea (entre artículos)
    antes que separadores de párrafo genéricos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\nArt.", "\nArtículo", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks, reset: bool = False):
    """Crea o actualiza el vector store en ChromaDB."""
    if reset and CHROMA_DIR.exists():
        print(f"  🗑️  Eliminando vector store anterior en: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    # OpenAI embeddings — text-embedding-3-small es económico y preciso en español
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Indexar documentos del SRI en ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Eliminar y re-crear el vector store")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Falta OPENAI_API_KEY en el archivo .env")
        sys.exit(1)

    print("\n📂 Estructura de carpetas esperada:")
    for folder, label in SUBFOLDERS.items():
        status = "✅" if (DOCS_DIR / folder).exists() else "❌ falta"
        print(f"   {status}  data/docs/{folder}/   ← {label}")

    print("\n🔍 Cargando documentos...")
    docs = load_documents()

    if not docs:
        print("\n❌ No se encontraron documentos. Crea las carpetas y agrega tus PDFs.")
        print("   Ejemplo: data/docs/lrti/LRTI_2024.pdf")
        sys.exit(1)

    print(f"\n✂️  Dividiendo en chunks (tamaño={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(docs)
    print(f"   → {len(docs)} páginas → {len(chunks)} chunks")

    print("\n🧠 Generando embeddings y guardando en ChromaDB...")
    build_vectorstore(chunks, reset=args.reset)

    print(f"\n✅ Listo. {len(chunks)} chunks indexados en: {CHROMA_DIR}")
    print("   Ahora puedes ejecutar: streamlit run scripts/app.py\n")


if __name__ == "__main__":
    main()
