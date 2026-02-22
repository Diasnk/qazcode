# Datasaur 2026 | Qazcode Challenge

## Medical Diagnosis Assistant: Symptoms → ICD-10

An AI-powered clinical decision support system that converts patient symptoms into structured diagnoses with ICD-10 codes, built on Kazakhstan clinical protocols.

---

## Challenge Overview

Participants will build an MVP product where users input symptoms as free text and receive:

- **Top-N probable diagnoses** ranked by likelihood
- **ICD-10 codes** for each diagnosis
- **Brief clinical explanations** based on official Kazakhstan protocols

The solution **must** run **using GPT-OSS** — no external LLM API calls allowed. Refer to `notebooks/llm_api_examples.ipynb`

---
## Data Sources

### Kazakhstan Clinical Protocols
Official clinical guidelines serving as the primary knowledge base for diagnoses and diagnostic criteria.[[corpus.zip](https://github.com/user-attachments/files/25365231/corpus.zip)]

Data Format

```json
{"protocol_id": "p_d57148b2d4", "source_file": "HELLP-СИНДРОМ.pdf", "title": "Одобрен", "icd_codes": ["O00", "O99"], "text": "Одобрен Объединенной комиссией по качеству медицинских услуг Министерства здравоохранения Республики Казахстан от «13» января 2023 года Протокол №177 КЛИНИЧЕСКИЙ ПРОТОКОЛ ДИАГНОСТИКИ И ЛЕЧЕНИЯ HELLP-СИНДРОМ I. ВВОДНАЯ ЧАСТЬ 1.1 Код(ы) МКБ-10: Код МКБ-10 O00-O99 Беременность, роды и послеродовой период О14.2 HELLP-синдром 1.2 Дата разработки/пересмотра протокола: 2022 год. ..."}

```

---

## Evaluation

### Metrics
- **Primary metrics:** Accuracy@1, Recall@3, Latency
- **Test set:**: Dataset with cases (`data/test_set`), use `query` and `gt` fields.
- **Holdout set:** Private test cases (not included in this repository)

### Product Evaluation
Working demo interface: user inputs symptoms → system returns diagnoses with ICD-10 codes;

---
## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/dair-mus/qazcode-nu.git
cd qazcode-nu
```

### 2. Set up the environment
We kindly ask you to use `uv` as your Python package manager.

Make sure that `uv` is installed. Refer to [uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3. Running validation
You can use `src/mock_server.py` as an example service. (however, it has no web UI, only an endpoint for eval). 
```bash
uv run uvicorn src.mock_server:app --host 127.0.0.1 --port 8000
```
Then run the validation pipeline in a separate terminal:
```bash
uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n <your_team_name>
```
`-e`: endpoint (POST request) that will accept the symptoms

`-d`: path to the directory with protocols

`-n`: name of your team (please avoid special symbols)

By default, the evalutaion results will be output to `data/evals`.

### Evaluating the notebook agent (vector store + LLM)

To run the same pipeline as in `notebooks/llm_api_examples.ipynb` (top-3 retrieval + GPT-OSS) and evaluate it:

1. **Prerequisites:** Build the vector store once in the notebook (load docs, add to Chroma). Ensure `.env` has `API_KEY` and `HUB_URL`. Start local embeddings: `docker compose up -d`.

2. **Start the diagnose server** (uses the notebook agent; reads Chroma from `notebooks/chroma_langchain_db`):
   ```bash
   uv run uvicorn src.diagnose_server:app --host 127.0.0.1 --port 8000
   ```

3. **Run the evaluator** in another terminal:
   ```bash
   uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n my_team
   ```

Results are written to `data/evals` as usual.

### Docker
We prepared a Dockerfile to run our mock server example.
```bash
docker build -t mock-server .
docker run -p 8000:8000 mock-server
```
Then run the validation as shown above.

Feel free to use the mock-server FastAPI template and Dockerfile structure to build your own project around.

Remember to adjust the CMD in Dockerfile for your real Python server instead of `src.mock_server:app` before submission. 

### Local Embeddings Model (Docker Compose)

The project includes a `docker-compose.yml` that runs a local embedding model alongside the app. The default is **ai-forever/ru-en-RoSBERTa** (Russian-focused, 768-dim), served by a lightweight FastAPI server (see `src/embedding_server.py`) that works on both ARM (Apple Silicon) and x86.

#### Run the diagnose server locally with Docker

1. **Create `.env`** (copy from `.env.example`) and set your OpenAI key:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-...
   ```

2. **Chroma DB (required):** Either build it once via the notebook (`notebooks/chroma_langchain_db`), or put `extracted_data/*.json` (with `identified_symptoms` and `gt`) in `./extracted_data` and set in `.env`:
   ```bash
   BUILD_CHROMA_IF_MISSING=1
   ```

3. **Start both services** (app + embeddings):
   ```bash
   docker compose up --build
   ```
   First run downloads the embedding model into `./models/` and may take a few minutes. The diagnose API is at **http://localhost:8000/diagnose**.

4. **Test:**
   ```bash
   uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n my_team
   ```

**Configuration:** Set `EMBEDDING_MODEL` in the embeddings service (e.g. in `docker-compose.yml` or a `.env` passed to the service) to use a different model. The server applies ru-en-RoSBERTa-style prefixes automatically: single-string `input` is treated as a query (`search_query:`), list of strings as documents (`search_document:`).

**Services:**
| Service | Port | Description |
|---|---|---|
| `app` | 8000 | Diagnose server (`src.diagnose_server_new`) — POST /diagnose |
| `embeddings` | 8081 | ru-en-RoSBERTa embedding API (OpenAI-compatible) |

**Usage example:**
```bash
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "У пациента головная боль", "model": "ai-forever/ru-en-RoSBERTa"}'
```

The endpoint is OpenAI-compatible:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="unused")
response = client.embeddings.create(
    input="У пациента головная боль",
    model="ai-forever/ru-en-RoSBERTa",
)
print(response.data[0].embedding)  # 768 dimensions
```

**Vector store (Chroma):** If you switch to ru-en-RoSBERTa from another model, rebuild the vector store because embedding dimensions and semantics change. Delete `notebooks/chroma_langchain_db` (or use a new collection/path), then re-run the notebook cells that load documents, split them, and call `vector_store.add_documents(...)` using the same embedding model.

### Submission Checklist

- [ ] Everything packed into a single project (application, models, vector DB, indexes)
- [ ] Image builds successfully: `docker build -t submission .`
- [ ] Container starts and serves on port 8080: `docker run -p 8080:8080 submission`
- [ ] Web UI accepts free-text symptoms input
- [ ] Endpoint for POST requests accepts free-text symptoms
- [ ] Returns top-N diagnoses with ICD-10 codes
- [ ] No external network calls during inference
- [ ] README with build and run instructions

### How to Submit

1. Provide a Git repository with `Dockerfile`
2. Submit the link via [submission form](https://docs.google.com/forms/d/e/1FAIpQLSe8qg6LsgJroHf9u_MVDBLPqD8S_W6MrphAteRqG-c4cqhQDw/viewform)
3. We will pull, build, and run your container on the private holdout set
---

### Repo structure
- `data/evals`: evaluation results directory
- `data/examples/response.json`: example of a JSON response from your project endpoint
- `data/test_set`: use these to evaluate your solution. 
- `notebooks/llm_api_examples.ipynb`: shows how to make a request to GPT-OSS.
- `src/`: solution source code would go here, has a `mock_server.py` as an entrypoint example.
- `evaluate.py`: runs the given dataset through the provided endpoint.
- `pyproject.toml`: describes dependencies of the project.
- `uv.lock`: stores the exact dependency versions, autogenerated by uv.
- `Dockerfile`: contains build instructions for a Docker image.
