import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Suppress "unauthenticated requests to HF Hub" warning — we run locally (model cached in /data)
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ru-en-RoSBERTa uses task prefixes: "search_query:" for queries, "search_document:" for docs
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "

DEFAULT_MODEL = "ai-forever/ru-en-RoSBERTa"
model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
app = FastAPI()
model = SentenceTransformer(model_name, cache_folder="/data")


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = DEFAULT_MODEL


def _prefix_texts(input_value: str | list[str]) -> list[str]:
    """Apply ru-en-RoSBERTa task prefix: single string = query, list = documents."""
    if isinstance(input_value, str):
        return [QUERY_PREFIX + input_value]
    return [DOCUMENT_PREFIX + t for t in input_value]


@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingRequest):
    texts = _prefix_texts(request.input)
    embeddings = model.encode(texts).tolist()
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(embeddings)
        ],
        "model": request.model,
    }
