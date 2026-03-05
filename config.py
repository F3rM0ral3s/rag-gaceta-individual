"""Configuración del backend RAG: rutas, URLs y parámetros de recuperación."""
from __future__ import annotations

import os
from pathlib import Path


def get_device() -> str:
    """Mejor dispositivo disponible para PyTorch: MPS (Apple Silicon) > CUDA > CPU.
    Se puede forzar con RAG_DEVICE=mps|cuda|cpu."""
    override = os.environ.get("RAG_DEVICE", "").strip().lower()
    if override in ("mps", "cuda", "cpu"):
        return override
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

# Base del repo
_rag_dir = Path(__file__).resolve().parent

# Embeddings en parquet
DEFAULT_PARQUET_PATH = os.environ.get(
    "RAG_PARQUET_PATH",
    str(_rag_dir / "data" / "embeddings"),
)
PARQUET_PATH = Path(DEFAULT_PARQUET_PATH)

# ChromaDB: persiste en rag/chroma_data por defecto; RAG_CHROMA_PATH="" para in-memory
DEFAULT_CHROMA_PATH = os.environ.get("RAG_CHROMA_PATH", str(_rag_dir / "chroma_data"))
CHROMA_PATH = Path(DEFAULT_CHROMA_PATH) if (DEFAULT_CHROMA_PATH and DEFAULT_CHROMA_PATH.strip()) else None

# llama-server (modelo GGUF)
LLAMA_BASE_URL = os.environ.get("RAG_LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_CHAT_ENDPOINT = f"{LLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"

# Modelo de embeddings (fijo)
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

# Recuperación: TOP_K fragmentos enviados al modelo; opcionalmente recupera más y conserva los mejores por distancia
TOP_K = int(os.environ.get("RAG_TOP_K", "10"))
TOP_K_RETRIEVE = int(os.environ.get("RAG_TOP_K_RETRIEVE", "0"))  # si > 0, recupera esa cantidad y conserva los mejores TOP_K por distancia

# Generación (llama-server) — menor = más factual
DEFAULT_MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.0"))

# Tamaño de lote al cargar parquet en ChromaDB
INGEST_BATCH_SIZE = int(os.environ.get("RAG_INGEST_BATCH_SIZE", "5000"))
