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

## Evaluation

The evaluation scripts measure **Accuracy@1** and **Recall@3** against a test set of clinical protocols. Each protocol JSON in the test set contains a patient query, ground-truth ICD-10 code, and valid ICD-10 codes. The scripts send each query to `POST /diagnose`, compare the returned top-3 codes against the ground truth, and produce per-protocol results (JSONL) plus aggregated metrics (JSON).

### Prerequisites

Make sure all three Docker services are running before you start:

```bash
docker compose up --build
```

Wait until you see `Diagnose server (notebook agent) running at http://127.0.0.1:8000/diagnose` in the logs — the API needs to load the Chroma index and connect to the LLM before it can serve requests.

### Full evaluation (`evaluate.py`)

Runs against **all** protocols in the test set. Execute from the `fastapi/` directory:

```bash
cd fastapi

uv run python evaluate.py \
  -e http://localhost:8000/diagnose \
  -d ./data/test_set \
  -n my_run \
  -p 4 \
  -o ./evals
```

| Flag | Long form | Description | Default |
|------|-----------|-------------|---------|
| `-e` | `--endpoint` | URL of the `/diagnose` endpoint (required) | — |
| `-d` | `--dataset-dir` | Directory with protocol JSON files (required) | — |
| `-n` | `--name` | Submission name, used for output filenames (required) | — |
| `-p` | `--parallelism` | Number of concurrent requests | `2` |
| `-o` | `--output-dir` | Where to write results | `data/evals` |

### Quick sanity check (`small_evaluate.py`)

Same interface as `evaluate.py` but limits to the first **N** protocols (default 5) so you can iterate quickly without waiting for the full set:

```bash
cd fastapi

uv run python small_evaluate.py \
  -e http://localhost:8000/diagnose \
  -d ./data/test_set \
  -n my_run \
  -p 4 \
  -o ./evals \
  -v
```

It accepts two extra flags on top of the ones above:

| Flag | Long form | Description | Default |
|------|-----------|-------------|---------|
| `-l` | `--limit` | Number of protocols to evaluate | `5` |
| `-v` | `--verbose` | Enable debug logging (full tracebacks on errors) | off |

For example, to evaluate only the first 20 protocols with verbose output:

```bash
uv run python small_evaluate.py \
  -e http://localhost:8000/diagnose \
  -d ./data/test_set \
  -n quick_check \
  -p 4 \
  -o ./evals \
  -l 20 \
  -v
```

### Output

After a run, two files appear in the output directory (e.g. `./evals`):

| File | Contents |
|------|----------|
| `<name>.jsonl` | One JSON object per protocol: response payload, accuracy, recall, latency, ground truth vs. prediction. |
| `<name>_metrics.json` | Aggregated metrics: Accuracy@1 %, Recall@3 %, latency stats (avg, min, max, p50, p95). |
