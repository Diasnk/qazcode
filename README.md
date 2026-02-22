# Project

## What runs locally

| Service    | Port | Description |
|-----------|------|-------------|
| **Embeddings** | 8081 | Local embedding model: **ai-forever/ru-en-RoSBERTa** (OpenAI-compatible `/v1/embeddings`). Model files are cached in `fastapi/models`. |
| **API (FastAPI)** | 8000 | RAG diagnostic server (`diagnose_server_new`): Chroma + LLM, exposes `POST /diagnose`. |
| **Website (Next.js)** | 3000 | Frontend app. |

## Setup

1. **Create `fastapi/.env`** with at least:
   ```bash
   OPENAI_API_KEY=your_key
   ```
   Optional: `OPENAI_MODEL` (default `gpt-4.1`).

2. **Chroma DB**: The API uses `fastapi/notebooks/chroma_langchain_db`, which already contains the vector embeddings of our normalized data. `BUILD_CHROMA_IF_MISSING=1` in `.env` is only a safety fallback to rebuild from `extracted_data` if the DB is missing—we don’t rely on it for normal runs.

3. **Start everything** (from project root):
   ```bash
   docker compose up
   ```
   - Website: http://localhost:3000  
   - API: http://localhost:8000  
   - Embeddings: http://localhost:8081 (used by the API internally).
