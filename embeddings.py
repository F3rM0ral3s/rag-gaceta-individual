"""Embeddings de consulta para RAG (deben coincidir con los embeddings del parquet)."""
from __future__ import annotations

import logging
from typing import List

from config import EMBEDDING_MODEL_ID, get_device

LOG = logging.getLogger(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        device = get_device()
        LOG.info("Cargando modelo de embeddings en el dispositivo: %s", device)
        _model = SentenceTransformer(EMBEDDING_MODEL_ID, device=device)
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    """Calcula embeddings para uno o más textos y devuelve vectores."""
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=False)
    return vectors.tolist()
