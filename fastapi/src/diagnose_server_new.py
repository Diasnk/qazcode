"""
RAG-based diagnostic server: same pipeline as notebooks/test2.ipynb
(Chroma + get_top3_protocols + LLM) exposed as POST /diagnose for evaluate.py.

Prerequisites:
  - .env: OPENAI_API_KEY (for LLM). Optional: OPENAI_MODEL (default gpt-4.1).
  - Embeddings service on EMBEDDING_BASE_URL (default http://localhost:8081/v1).
  - Vector store: either run the notebook once to build notebooks/chroma_langchain_db,
    or set BUILD_CHROMA_IF_MISSING=1 and have extracted_data/*.json (with
    identified_symptoms + gt) so the server builds the DB on startup.

Env:
  - OPENAI_API_KEY, OPENAI_MODEL (LLM)
  - EMBEDDING_BASE_URL (default http://localhost:8081/v1)
  - EMBEDDING_MODEL (default ai-forever/ru-en-RoSBERTa)
  - BUILD_CHROMA_IF_MISSING=1 to build Chroma from extracted_data if DB missing
  - LOG_LEVEL=DEBUG for parse/retrieval details

Usage:
  uv run uvicorn src.diagnose_server_new:app --host 127.0.0.1 --port 8000

Test:
  uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n my_team
"""

import hashlib
import json
import logging
import os
import re
import traceback
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Request-scoped id for correlating logs (set at entry, read in predict/parse/tool)
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def _log(level: str, msg: str, *args, **kwargs) -> None:
    """Log with request_id prefix when set."""
    rid = _request_id.get()
    prefix = f"[{rid}] " if rid else ""
    getattr(logger, level)(prefix + msg, *args, **kwargs)

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ---------- Pydantic models (match evaluate.py expected format) ----------
class DiagnosisEntry(BaseModel):
    """One diagnosis from the agent: name, ICD-10 code, and short explanation."""

    diagnosis: str = ""
    icd10_code: str
    explanation: str = ""


class AgentResponse(BaseModel):
    """Agent can return full diagnoses (preferred) or only ICD-10 codes (fallback)."""

    diagnoses: list[DiagnosisEntry] | None = None
    ICD_10_code: Annotated[
        list[str] | None,
        Field(description="ICD-10 codes (legacy)", min_length=3, max_length=3),
    ] = None


class DiagnoseRequest(BaseModel):
    symptoms: str | None = ""

    def get_symptoms(self) -> str:
        return (self.symptoms or "").strip() or ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ---------- Globals (set in lifespan) ----------
agent = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "notebooks" / "chroma_langchain_db"
DATA_DIR = PROJECT_ROOT / "extracted_data"
BATCH_SIZE = 1000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "example_collection"
RETRIEVAL_K = 3


def _message_content_to_str(content) -> str:
    """Normalize message content to string; handle list of content blocks (e.g. OpenAI)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", block.get("content", str(block))))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


# ICD-10 pattern: letter + 2 digits, optional . + digits (e.g. S22.0, R69, G43.1)
_ICD10_PATTERN = re.compile(r"\b([A-Z]\d{2}(?:\.\d+)?)\b", re.IGNORECASE)


def _parse_agent_response(content: str) -> AgentResponse:
    if not (content or content.strip()):
        _log("warning", "Parse failed: empty agent response")
        raise ValueError("Empty agent response")
    raw = content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw).rstrip("`").strip()
    try:
        out = AgentResponse.model_validate_json(raw)
        _log("debug", "Parsed ICD codes via full JSON: %s", out.ICD_10_code)
        return out
    except Exception as e1:
        _log("debug", "Full JSON parse failed: %s", e1)
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(raw[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            if depth == 0:
                try:
                    out = AgentResponse.model_validate_json(raw[start : i + 1])
                    _log("debug", "Parsed ICD codes via bracket extraction: %s", out.ICD_10_code)
                    return out
                except Exception as e2:
                    _log("debug", "Bracket JSON parse failed: %s", e2)
                    break
    for key in ("ICD_10_code", "ICD_10_codes", "icd10_code", "icd_10_code"):
        m = re.search(
            rf'"{key}"\s*:\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*',
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            codes = [m.group(1), m.group(2), m.group(3)]
            _log("debug", "Parsed ICD codes via regex (%s): %s", key, codes)
            return AgentResponse(ICD_10_code=codes)
    # Relaxed: two or three quoted codes in array (any key)
    m = re.search(r'\[\s*"([A-Z]\d{2}(?:\.\d+)?)"\s*,\s*"([A-Z]\d{2}(?:\.\d+)?)"\s*(?:,\s*"([A-Z]\d{2}(?:\.\d+)?)")?\s*\]', raw, re.IGNORECASE)
    if m:
        codes = [m.group(1), m.group(2), m.group(3) or "R69"]
        if len(codes) < 3:
            codes.extend(["R69"] * (3 - len(codes)))
        _log("debug", "Parsed ICD codes via array regex: %s", codes[:3])
        return AgentResponse(ICD_10_code=codes[:3])
    # Last resort: find up to 3 ICD-10-like codes in text (order preserved)
    found = list(dict.fromkeys(_ICD10_PATTERN.findall(raw)))  # unique, order preserved
    if len(found) >= 1:
        codes = (found + ["R69", "R69"])[:3]
        _log("debug", "Parsed ICD codes via pattern fallback: %s", codes)
        return AgentResponse(ICD_10_code=codes)
    _log(
        "warning",
        "Parse failed: no ICD_10_code found in response (first 500 chars): %s",
        raw[:500],
    )
    raise ValueError(f"Could not parse ICD_10_code from response: {raw[:500]}")


def _agent_response_to_entries(parsed: AgentResponse) -> list[DiagnosisEntry]:
    """Normalize Agent"падение с отведенной рукой и удар плечом",
    "боль в области соединения ключицы с грудью (центр кости)",
    "ограничение поднятия руки вверх, боль при попытке надеть одежду или дотянуться до верхних полок",
    "ощущается отёк и болезненная шишка в зоне ключицы",
    "ощущение щёлкающего звука/перемещения при движении",
    "ночная боль, невозможность удобно лечь на поражённый бок",
    "боль при глубоком вдохе"Response to exactly 3 DiagnosisEntry items for the API."""
    if parsed.diagnoses and len(parsed.diagnoses) >= 3:
        return list(parsed.diagnoses[:3])
    codes = list((parsed.ICD_10_code or [])[:3])
    while len(codes) < 3:
        codes.append("R69")
    return [
        DiagnosisEntry(diagnosis="", icd10_code=c, explanation="")
        for c in codes
    ]


def _build_chroma_from_extracted_data(
    embeddings,
    persist_dir: Path,
    data_dir: Path,
) -> None:
    """Load extracted_data/*.json, split, and add to Chroma (same as notebook)."""
    docs = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        symptoms = data.get("identified_symptoms", [])
        page_content = "\n".join(symptoms) if isinstance(symptoms, list) else str(symptoms)
        docs.append(
            Document(
                page_content=page_content,
                metadata={"gt": data.get("gt", "")},
            )
        )
    if not docs:
        raise RuntimeError(f"No documents loaded from {data_dir}. Need *.json with identified_symptoms and gt.")
    logger.info("Loaded %d documents from %s", len(docs), data_dir)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    all_splits = docs

    persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    for i in range(0, len(all_splits), BATCH_SIZE):
        batch = all_splits[i : i + BATCH_SIZE]
        vector_store.add_documents(documents=batch)
        logger.info("Added batch %d (%d docs)", (i // BATCH_SIZE) + 1, len(batch))
    logger.info("Chroma DB built at %s with %d chunks", persist_dir, len(all_splits))


def predict(symptoms: str) -> AgentResponse:
    """Run the agent and return structured ICD-10 codes."""
    _log("info", "Agent invoke starting (symptoms length=%d)", len(symptoms))
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": symptoms}]})
    except Exception as e:
        _log(
            "warning",
            "Agent invoke failed: %s: %s",
            type(e).__name__,
            e,
            exc_info=True,
        )
        raise
    messages = result.get("messages") or []
    _log("info", "Agent returned %d messages", len(messages))
    # Normalize all message contents to string (content can be list of blocks)
    content = ""
    for m in reversed(messages):
        c = _message_content_to_str(getattr(m, "content", None))
        if not c.strip():
            continue
        # Prefer content that looks like final answer (JSON with diagnoses or codes)
        if "diagnoses" in c or "ICD_10" in c or "icd_10" in c or ("{" in c and ("code" in c.lower() or "[" in c)):
            content = c
            break
    if not content and messages:
        # Fallback: last message with any non-empty content
        for m in reversed(messages):
            c = _message_content_to_str(getattr(m, "content", None))
            if c.strip():
                content = c
                break
    if not content.strip():
        _log(
            "warning",
            "No usable content in agent messages (last message type: %s)",
            type(messages[-1]).__name__ if messages else "none",
        )
    else:
        _log(
            "debug",
            "Using response snippet: %s...",
            (content.strip()[:200] + "..." if len(content) > 200 else content.strip()),
        )
    return _parse_agent_response(content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env for the LLM.")

    embedding_base = os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8081/v1")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "ai-forever/ru-en-RoSBERTa")
    embedding_key = os.environ.get("EMBEDDING_API_KEY", "unused")

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_base=embedding_base,
        openai_api_key=embedding_key,
        check_embedding_ctx_length=False,
    )

    if not CHROMA_DIR.exists():
        build_if_missing = os.environ.get("BUILD_CHROMA_IF_MISSING", "").strip() in ("1", "true", "yes")
        if build_if_missing and DATA_DIR.exists():
            logger.info("Building Chroma from %s -> %s", DATA_DIR, CHROMA_DIR)
            _build_chroma_from_extracted_data(embeddings, CHROMA_DIR, DATA_DIR)
        else:
            raise RuntimeError(
                f"Chroma DB not found at {CHROMA_DIR}. "
                "Run the notebook once to build it, or set BUILD_CHROMA_IF_MISSING=1 and ensure "
                f"extracted_data exists at {DATA_DIR}."
            )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    @tool
    def get_top3_protocols(query: str) -> str:
        """Retrieve the top 3 most relevant clinical protocol excerpts for the given patient symptoms query. Call this first with the user's symptoms."""
        _log("info", "Retrieval query length=%d, k=%d", len(query), RETRIEVAL_K)
        try:
            results = vector_store.similarity_search_with_score(query, k=RETRIEVAL_K)
        except Exception as e:
            _log(
                "warning",
                "Retrieval failed: %s: %s",
                type(e).__name__,
                e,
                exc_info=True,
            )
            raise
        scores = [s for _, s in results]
        _log(
            "info",
            "Retrieval returned %d docs (scores min=%.3f max=%.3f)",
            len(results),
            min(scores) if scores else 0,
            max(scores) if scores else 0,
        )
        parts = []
        for i, (doc, score) in enumerate(results, 1):
            parts.append(
                f"[Match {i}, score={score:.3f}]\n"
                f"GT_ICD: {doc.metadata.get('gt')}\n"
                f"Symptoms:\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    model_id = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    llm = init_chat_model(model_id, temperature=0)

    prompt = """
        You are a clinical decision support system.

        You MUST follow this workflow exactly:

        1) Call `get_top3_protocols` first.
        2) Use ONLY the retrieved cases/protocols as evidence.
        3) Extract the patient's symptoms into a normalized list (canonical clinical terms).
        4) Compare the patient symptoms to EACH retrieved case separately.
        5) For each retrieved case, identify:
        - matching symptoms (explicit overlap)
        - missing expected symptoms
        - contradictory symptoms (if any)
        - demographic/context matches (age/sex if available)
        6) Aggregate evidence across all retrieved cases by ICD-10 code.
        7) Rank diagnoses using this priority order:
        a) Highest symptom-overlap score
        b) ICD-10 code frequency across retrieved cases (prefer codes repeated multiple times)
        c) Fewer contradictions
        d) Better match to demographics/context
        8) Output exactly 3 UNIQUE ICD-10 codes ranked by likelihood.

        Scoring guidance (use internally):
        - Strong symptom match: +2
        - Supporting symptom match: +1
        - Demographic/context match: +1
        - Contradictory symptom: -2
        - Missing hallmark symptom: -1

        Rules:
        - Use ONLY retrieved content
        - Do NOT invent diagnoses or codes not present in retrieved cases
        - If uncertain, still choose the closest 3 ICD-10 codes from retrieved content
        - Explanations must mention the specific overlapping symptoms and whether the code appeared in multiple retrieved cases
        - Output JSON only (no markdown, no extra text)

        Required JSON format:
        {
        "diagnoses": [
            {
            "rank": 1,
            "diagnosis": "",
            "icd10_code": "",
            "explanation": ""
            },
            {
            "rank": 2,
            "diagnosis": "",
            "icd10_code": "",
            "explanation": ""
            },
            {
            "rank": 3,
            "diagnosis": "",
            "icd10_code": "",
            "explanation": ""
            }
        ]
        }
        """
    
    agent = create_agent(llm, [get_top3_protocols], system_prompt=prompt)

    print("\nDiagnose server (notebook agent) running at http://127.0.0.1:8000/diagnose")
    yield
    print("\nShutdown.")


app = FastAPI(title="Diagnostic Server (Agent)", lifespan=lifespan)

# CORS so browser clients (e.g. Next.js) can call /diagnose (handles OPTIONS preflight)
_allowed_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

FALLBACK_RESPONSE = DiagnoseResponse(
    diagnoses=[
        Diagnosis(rank=1, diagnosis="Unspecified", icd10_code="R69", explanation="Diagnosis could not be determined."),
        Diagnosis(rank=2, diagnosis="Unspecified", icd10_code="R69", explanation="Diagnosis could not be determined."),
        Diagnosis(rank=3, diagnosis="Unspecified", icd10_code="R69", explanation="Diagnosis could not be determined."),
    ]
)


@app.exception_handler(Exception)
async def catch_all(request, exc: Exception):
    """Log any unhandled exception and return 200 with fallback so evaluation completes."""
    rid = _request_id.get() or "no-rid"
    logger.exception("[%s] Unhandled exception in /diagnose: %s", rid, exc)
    return JSONResponse(
        status_code=200,
        content=FALLBACK_RESPONSE.model_dump(),
        media_type="application/json",
    )


def _request_id_from_symptoms(symptoms: str) -> str:
    """Stable short id for correlating logs (same symptoms => same id)."""
    if not symptoms:
        return "empty"
    h = hashlib.sha256(symptoms.encode("utf-8")[:500]).hexdigest()
    return h[:12]


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    """Run the agent on symptoms and return top 3 ICD-10 codes in evaluate.py format."""
    symptoms = request.get_symptoms()
    request_id = _request_id_from_symptoms(symptoms)
    token = _request_id.set(request_id)

    preview = (symptoms[:80] + "…") if len(symptoms) > 80 else symptoms
    _log(
        "info",
        "Request start | symptoms_len=%d preview=%r",
        len(symptoms),
        preview or "(empty)",
    )

    try:
        parsed = predict(symptoms)
        entries = _agent_response_to_entries(parsed)
        _log(
            "info",
            "Request success | codes=%s",
            [e.icd10_code for e in entries],
        )
    except Exception as e:
        _log(
            "warning",
            "Request fallback (R69) | reason=%s: %s",
            type(e).__name__,
            e,
        )
        _log("debug", "Traceback: %s", traceback.format_exc())
        entries = [
            DiagnosisEntry(diagnosis="Unspecified", icd10_code="R69", explanation="Diagnosis could not be determined.")
            for _ in range(3)
        ]
    finally:
        _request_id.reset(token)

    while len(entries) < 3:
        entries.append(DiagnosisEntry(diagnosis="Unspecified", icd10_code="R69", explanation=""))
    diagnoses = [
        Diagnosis(rank=i, diagnosis=e.diagnosis, icd10_code=e.icd10_code, explanation=e.explanation)
        for i, e in enumerate(entries[:3], start=1)
    ]
    return DiagnoseResponse(diagnoses=diagnoses)
