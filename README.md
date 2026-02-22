# Project

RAG-based diagnostic app: a **local embedding model** (RoSBERTa) produces vectors, **Chroma** holds the pre-built index, and **LangChain** wires retrieval + an LLM agent. Chat completions use the **QazCode hub** (OpenAI-compatible API). The **Next.js** frontend talks to the FastAPI backend.

## Services (run from repo root)

| Service | Port | Description |
|--------|------|-------------|
| **Embeddings** | 8082 | Local embedding model (ai-forever/ru-en-RoSBERTa), OpenAI-compatible `/v1/embeddings`. Cached in `fastapi/models`. |
| **API (FastAPI)** | 8000 | RAG diagnostic server: Chroma + QazCode LLM, `POST /diagnose`. |
| **Web (Next.js)** | 3000 | Frontend. |

## Setup

1. **Env** — Copy `fastapi/.env.example` → `fastapi/.env` and set:
   - `API_KEY` — QazCode hub API key (OpenAI-compatible).
   - `HUB_URL` — QazCode hub base URL (e.g. `https://hub.qazcode.ai`).
   - `EMBEDDING_MODEL` — (optional) Hugging Face model for local embeddings; default `ai-forever/ru-en-RoSBERTa`.

2. **Chroma** — The repo ships a pre-built Chroma DB at `fastapi/notebooks/chroma_langchain_db` (metadata + HNSW index). No need to run notebooks or re-embed; the API uses it as-is. To rebuild from `extracted_data`, set `BUILD_CHROMA_IF_MISSING=1` in `.env` (fallback only).

3. **Build & run** (project root):
   ```bash
   docker compose build
   docker compose up
   ```
   Or in one step: `docker compose up --build`.
   - Web: http://localhost:3000  
   - API: http://localhost:8000  
   - Embeddings: http://localhost:8082 (used by the API).
