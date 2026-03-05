# RAG Personal (ChromaDB + llama-server)

RAG para Q&A sobre embeddings generados a partir de GACETA UNAM. Incluye un backend en FastAPI, ChromaDB para la búsqueda por similitud y una UI en Streamlit.

Puede usarse para otro tipo de embeddings pero se necesitaría generarlos el usuario y cambiar las confirguraciones de vectorización que estan fijos a BGE-M3.

## Requisitos

- Python 3.10+
- `llama-server` corriendo con un modelo **GGUF**, expuesto en `/v1/chat/completions`
  - Ejemplo:
    ```bash
    llama-server -m /path/to/your-model.gguf --port 8080
    ```

## Instalación

```bash
cd rag
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

## Configuración


Crea un archivo  `.env`  y setea tus variables de entorno

- `RAG_PARQUET_PATH` – ruta al parquet o directorio de parquets (por defecto: `data/embeddings`).
- `RAG_CHROMA_PATH` – directorio persistente de ChromaDB (por defecto: `chroma_data`). Deja vacío para in-memory.
- `RAG_LLAMA_BASE_URL` – URL base del `llama-server` (por defecto: `http://127.0.0.1:8080`).
- El modelo de embeddings es fijo: `BAAI/bge-m3`.
- `RAG_DEVICE` – `mps|cuda|cpu` para forzar el dispositivo de embeddings.
- `RAG_TOP_K` – número de fragmentos enviados al modelo (por defecto: 10).
- `RAG_TOP_K_RETRIEVE` – si > 0, recupera esa cantidad y conserva los mejores TOP_K por distancia.
- `RAG_TEMPERATURE` – temperatura de generación (por defecto: 0.0).
- `RAG_MAX_TOKENS` – máximo de tokens de salida (por defecto: 2048).
- `RAG_BACKEND_URL` – URL del backend para Streamlit (por defecto: `http://127.0.0.1:8000`).

## Ejecución

0. **Carga de datos** 
Descarga el parquet de Gaceta UNAM en:

Colocalo en '''/chroma_data'''

1. **Backend** (carga parquet en Chroma si la colección está vacía):

   ```bash
   cd rag
   source .venv/bin/activate
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. **Streamlit UI** (en otra terminal):

   ```bash
   cd rag
   source .venv/bin/activate
   streamlit run streamlit_ui.py
   ```

   Si `streamlit` no está en PATH:
   `./.venv/bin/python -m streamlit run streamlit_ui.py`

3. Abre la URL que muestra Streamlit (por ejemplo `http://localhost:8501`) y escribe tu pregunta.

## Personalización

- Prompt del sistema: edita `RAG_SYSTEM_PROMPT` en `app.py`.
- Columnas de metadata: ajusta `METADATA_COLUMNS` en `chroma_store.py` para que coincida con tu esquema.
