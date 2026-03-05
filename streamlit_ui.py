"""UI en Streamlit para RAG: chat, llamada a /chat, respuesta y fuentes."""
from __future__ import annotations

import os
import streamlit as st
import httpx

BACKEND_URL = os.environ.get("RAG_BACKEND_URL", "http://127.0.0.1:8000")
CHAT_URL = f"{BACKEND_URL.rstrip('/')}/chat"

st.set_page_config(page_title="RAG Personal", page_icon="📄", layout="centered")
st.title("RAG Personal")
st.caption("Pregunta sobre tu corpus. Respuestas generadas con el contexto recuperado.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Fuentes"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**Fragmento {i}**")
                    st.text(s.get("text", "")[:500] + ("..." if len(s.get("text", "")) > 500 else ""))
                    if s.get("metadata"):
                        st.caption(str(s["metadata"]))

if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            try:
                # Sin timeout explícito: lo decide el backend / llama-server
                r = httpx.post(CHAT_URL, json={"req": prompt}, timeout=None)
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
            except httpx.ConnectError:
                answer = "No se pudo conectar con el backend. Asegúrate de que el servidor esté corriendo (por ejemplo: `uvicorn app:app --reload`)."
                sources = []
            except httpx.HTTPStatusError as e:
                answer = f"Error del servidor: {e.response.status_code}. {e.response.text[:200]}"
                sources = []
            except Exception as e:
                answer = f"Error: {str(e)}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Fuentes utilizadas"):
                for i, s in enumerate(sources, 1):
                    st.markdown(f"**Fragmento {i}**")
                    text = s.get("text", "")
                    st.text(text[:600] + ("..." if len(text) > 600 else ""))
                    meta = s.get("metadata", {})
                    if meta:
                        st.caption(", ".join(f"{k}: {v}" for k, v in meta.items()))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
