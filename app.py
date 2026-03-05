"""Backend FastAPI: RAG sobre ChromaDB + llama-server local."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    LLAMA_CHAT_ENDPOINT,
    PARQUET_PATH,
    TOP_K,
    TOP_K_RETRIEVE,
)
from chroma_store import ensure_collection, get_client, query_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOG = logging.getLogger(__name__)

# Referencia global a la colección de Chroma (se configura al iniciar)
_collection = None


def _get_collection():
    if _collection is None:
        raise RuntimeError("Chroma collection not initialized")
    return _collection


RAG_SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas usando exclusivamente los fragmentos de contexto proporcionados.\n\n"
    "## REGLAS ESTRICTAS (obligatorias)\n"
    "1. Usa ÚNICAMENTE la información que aparece en los fragmentos. No uses conocimiento externo.\n"
    "2. Si la respuesta no está en los fragmentos, responde exactamente: \"No encontré esa información en los fragmentos proporcionados.\" No inventes ni sugieras datos.\n"
    "3. No inventes fechas, nombres, cifras, normativas ni datos. No deduzcas ni infieras cosas que no estén explícitas en el texto.\n"
    "4. Responde en español, de forma clara, directa y concisa.\n"
    "5. Cita cada afirmación con el número del fragmento entre corchetes, ej: [1], [2]. Si hay metadata (documento, fecha, páginas), inclúyela para que el usuario verifique la fuente.\n\n"
    "## FORMATO Y ESTILO\n"
    "- Da respuestas breves y al punto; evita rodeos.\n"
    "- Si la pregunta tiene varias partes, responde por partes o en un párrafo ordenado.\n"
    "- Si solo encuentras información parcial, di lo que sí está en los fragmentos y aclara qué no encontraste.\n"
    "- No repitas la pregunta del usuario; ve directo a la respuesta.\n"
    "- No inventes \"según el fragmento X\" si no estás citando algo que realmente dice ese fragmento."
)


def build_user_prompt(
    context_chunks: list[str],
    question: str,
    metadatas: list[dict] | None = None,
) -> str:
    """Construye el prompt con fragmentos numerados y metadata opcional."""
    metadatas = metadatas or [{}] * len(context_chunks)
    parts = ["Fragmentos de contexto (usa solo esta información para responder):\n"]
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadatas), 1):
        label_parts = []
        if meta.get("issue_date"):
            label_parts.append(f"fecha: {meta['issue_date']}")
        if meta.get("source_pdf"):
            label_parts.append(f"doc: {meta['source_pdf']}")
        if meta.get("page_start") is not None and meta.get("page_end") is not None:
            label_parts.append(f"págs. {meta['page_start']}-{meta['page_end']}")
        label = f" ({', '.join(label_parts)})" if label_parts else ""
        parts.append(f"[Fragmento {i}{label}]\n{chunk}\n")
    parts.append("\nPregunta del usuario:\n")
    parts.append(question)
    return "\n".join(parts)


async def call_llama(messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(LLAMA_CHAT_ENDPOINT, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected llama-server response: {data}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _collection
    client = get_client()
    _collection = ensure_collection(client, parquet_path=PARQUET_PATH)
    LOG.info("RAG backend ready. Chroma collection: %s documents", _collection.count())
    yield
    _collection = None


app = FastAPI(title="RAG Backend", description="Consulta tu corpus vía ChromaDB + llama-server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    req: str


class SourceItem(BaseModel):
    text: str
    metadata: dict


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """RAG: recuperar contexto -> llamar al modelo -> devolver respuesta y fuentes."""
    req = body.req
    if not req or not req.strip():
        raise HTTPException(status_code=400, detail="message is required")
    coll = _get_collection()
    message = req.strip()

    n_retrieve = max(TOP_K, TOP_K_RETRIEVE) if TOP_K_RETRIEVE > 0 else TOP_K
    result = query_collection(
        coll,
        message,
        top_k=n_retrieve,
        top_k_retrieve=TOP_K_RETRIEVE if TOP_K_RETRIEVE > 0 else None,
    )

    documents = result["documents"] or []
    metadatas = result["metadatas"] or []

    sources = [SourceItem(text=doc, metadata=meta or {}) for doc, meta in zip(documents, metadatas)]
    user_prompt = build_user_prompt(documents, message, metadatas)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        answer = await call_llama(messages)
    except httpx.HTTPStatusError as e:
        LOG.exception("llama-server error")
        raise HTTPException(status_code=502, detail=f"llama-server error: {e.response.text}") from e
    except Exception as e:
        LOG.exception("llama-server request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {"status": "ok", "collection_count": _get_collection().count()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
