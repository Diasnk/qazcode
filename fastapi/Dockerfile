# Diagnose server: FastAPI app (src.diagnose_server_new) + Chroma + LLM.
# Expects embeddings service at EMBEDDING_BASE_URL; Chroma DB via volume or BUILD_CHROMA_IF_MISSING.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies (diagnose server + Chroma + LangChain)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Application code
COPY src/ src/

# Directories for mounted volumes (Chroma DB and optional extracted_data)
RUN mkdir -p /app/notebooks/chroma_langchain_db /app/extracted_data

EXPOSE 8000

CMD ["uvicorn", "src.diagnose_server_new:app", "--host", "0.0.0.0", "--port", "8000"]
