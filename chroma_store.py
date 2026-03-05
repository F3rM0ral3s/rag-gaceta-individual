"""Almacén ChromaDB: carga parquet con embeddings y ejecuta búsqueda por similitud."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import chromadb
from chromadb.config import Settings
import pyarrow.parquet as pq

from config import (
    CHROMA_PATH,
    INGEST_BATCH_SIZE,
    PARQUET_PATH,
    TOP_K,
    TOP_K_RETRIEVE,
)
from embeddings import embed

LOG = logging.getLogger(__name__)
COLLECTION_NAME = "rag_collection"

# La metadata en Chroma debe ser str, int, float o bool
METADATA_COLUMNS = ("doc_id", "chunk_id", "chunk_index", "corpus", "decade", "issue_date", "page_start", "page_end", "source_pdf", "source_file")


def _metadata_row(row: Any, column_names: List[str]) -> dict:
    out = {}
    for key in METADATA_COLUMNS:
        if key not in column_names:
            continue
        idx = column_names.index(key)
        val = row[idx]
        if val is None:
            continue
        if isinstance(val, (str, int, float, bool)):
            out[key] = val
        else:
            out[key] = str(val)
    return out


def _load_one_parquet(path: Path):
    """Devuelve lotes (ids, embeddings, documentos, metadatas) desde un parquet."""
    table = pq.read_table(path)
    column_names = table.column_names
    n = table.num_rows

    ids_col = table.column("chunk_id")
    text_col = table.column("text")
    emb_col = table.column("embedding")

    for start in range(0, n, INGEST_BATCH_SIZE):
        end = min(start + INGEST_BATCH_SIZE, n)
        ids = [ids_col[i].as_py() if hasattr(ids_col[i], "as_py") else str(ids_col[i]) for i in range(start, end)]
        documents = [text_col[i].as_py() if hasattr(text_col[i], "as_py") else str(text_col[i]) for i in range(start, end)]
        embeddings = []
        for i in range(start, end):
            e = emb_col[i]
            if hasattr(e, "as_py"):
                embeddings.append(e.as_py())
            else:
                embeddings.append(list(e))
        metadatas = []
        for i in range(start, end):
            row = [table.column(c)[i] for c in column_names]
            meta = _metadata_row(row, column_names)
            if "chunk_index" in meta and not isinstance(meta["chunk_index"], int):
                meta["chunk_index"] = int(meta["chunk_index"])
            metadatas.append(meta)
        yield ids, embeddings, documents, metadatas


def load_parquet_columns(path: Path):
    """Devuelve lotes (ids, embeddings, documentos, metadatas) desde uno o varios parquet.
    Si `path` es directorio, lee todos los .parquet (ordenados por nombre).
    """
    if path.is_dir():
        parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files in directory: {path}")
        for p in parquet_files:
            yield from _load_one_parquet(p)
    else:
        yield from _load_one_parquet(path)


def get_client():
    """Devuelve cliente de Chroma (persistente o en memoria)."""
    if CHROMA_PATH:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False))
    return chromadb.Client(Settings(anonymized_telemetry=False))


def ensure_collection(client, parquet_path: Optional[Path] = None):
    """Obtiene o crea la colección; si está vacía y hay parquet, lo carga."""
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Fragmentos con embeddings para RAG"},
    )
    path = parquet_path or PARQUET_PATH
    if path and path.exists() and coll.count() == 0:
        LOG.info("Cargando parquet en ChromaDB: %s", path)
        for ids, embeddings, documents, metadatas in load_parquet_columns(path):
            coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
            LOG.info("Lote agregado, total ahora %s", coll.count())
        LOG.info("Carga ChromaDB completa. Total documentos: %s", coll.count())
    return coll


def query_collection(
    collection: chromadb.Collection,
    message: str,
    top_k: int = TOP_K,
    top_k_retrieve: int | None = None,
) -> dict:
    """Embebe el mensaje, consulta ChromaDB y devuelve top_k. Si top_k_retrieve > top_k, recupera más y se queda con los mejores por distancia."""
    if top_k_retrieve is None:
        top_k_retrieve = TOP_K_RETRIEVE
    n = max(top_k, top_k_retrieve) if top_k_retrieve > 0 else top_k
    query_embeddings = embed([message])
    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    dists = result["distances"][0] if result.get("distances") else []
    if top_k_retrieve > 0 and len(docs) > top_k and dists:
        # Chroma usa distancia L2 (menor = mejor); ordenar por distancia y tomar top_k
        indexed = list(zip(dists, docs, metas))
        indexed.sort(key=lambda x: x[0])
        dists, docs, metas = zip(*indexed[:top_k])
        docs, metas = list(docs), list(metas)
    return {"documents": docs, "metadatas": metas, "distances": dists}
