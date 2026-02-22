"""
Diagnostic server that runs the notebook agent (get_top3_protocols + LLM)
and exposes POST /diagnose for evaluate.py.

Prerequisites:
  - .env with OPENAI_API_KEY (for LLM via init_chat_model). Optional: OPENAI_MODEL (default gpt-4.1, same as notebook).
  - Local embeddings: docker compose up -d (or embeddings on localhost:8081)
  - Vector store built: run the notebook once to populate notebooks/chroma_langchain_db

Usage:
  uv run uvicorn src.diagnose_server:app --host 127.0.0.1 --port 8000

Then:
  uv run python evaluate.py -e http://127.0.0.1:8000/diagnose -d ./data/test_set -n my_team

If Accuracy@1 and Recall@3 are 0%, check server logs: often every request hits the
fallback (R69) because the agent or parse failed. Ensure embeddings and LLM are
reachable, then re-run evaluation.

Logging: Each request gets a 12-char request_id (hash of symptoms) so you can grep
one protocol. Set LOG_LEVEL=DEBUG (e.g. env or logging.basicConfig) for parse details
and response snippets.
"""

import hashlib
import logging
import os
import re
import traceback
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from dotenv import load_dotenv

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
from pydantic import BaseModel, Field
from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# ---------- Pydantic models (match evaluate.py expected format) ----------
class AgentResponse(BaseModel):
    ICD_10_code: Annotated[
        list[str],
        Field(description="ICD-10 codes", min_length=3, max_length=3),
    ]


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
CHROMA_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "chroma_langchain_db"


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
        # Prefer content that looks like final answer (JSON with codes)
        if "ICD_10" in c or "icd_10" in c or ("{" in c and ("code" in c.lower() or "[" in c)):
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
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Chroma DB not found at {CHROMA_DIR}. "
            "Run the notebook once to build the vector store (load docs, add to Chroma)."
        )

    embeddings = OpenAIEmbeddings(
        model="ai-forever/ru-en-RoSBERTa",
        openai_api_base="http://localhost:8081/v1",
        openai_api_key="unused",
        check_embedding_ctx_length=False,
    )
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # Match notebook: top 3 protocol excerpts (get_top3_protocols uses k=3)
    RETRIEVAL_K = 3

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
                f"[Match {i}, score={score:.3f}] Source: {doc.metadata}\nContent: {doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    model_id = os.environ.get("OPENAI_MODEL", "gpt-4.1")
    llm = init_chat_model(model_id, temperature=0)
    prompt = (
        "You are a medical diagnosis assistant. "
        "You MUST call get_top3_protocols first with the patient's symptoms (the user message) to retrieve the top 3 relevant clinical protocol excerpts. "
        "Based only on that retrieved context, respond with exactly 3 ICD-10 codes (most probable diagnoses, ranked by likelihood). "
        "Your final response must be a valid JSON object with this exact structure: {\"ICD_10_code\": [\"code1\", \"code2\", \"code3\"]}. "
        "Example: {\"ICD_10_code\": [\"R42\", \"G43.1\", \"F41.0\"]}. No other text, only the JSON."
    )
    agent = create_agent(llm, [get_top3_protocols], system_prompt=prompt)

    print("\nDiagnose server (notebook agent) running at http://127.0.0.1:8000/diagnose")
    yield
    print("\nShutdown.")


app = FastAPI(title="Diagnostic Server (Agent)", lifespan=lifespan)

FALLBACK_RESPONSE = DiagnoseResponse(
    diagnoses=[
        Diagnosis(rank=1, diagnosis="", icd10_code="R69", explanation=""),
        Diagnosis(rank=2, diagnosis="", icd10_code="R69", explanation=""),
        Diagnosis(rank=3, diagnosis="", icd10_code="R69", explanation=""),
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
        codes = list(parsed.ICD_10_code[:3])
        _log(
            "info",
            "Request success | codes=%s",
            codes,
        )
    except Exception as e:
        _log(
            "warning",
            "Request fallback (R69) | reason=%s: %s",
            type(e).__name__,
            e,
        )
        _log("debug", "Traceback: %s", traceback.format_exc())
        codes = ["R69", "R69", "R69"]
    finally:
        _request_id.reset(token)

    while len(codes) < 3:
        codes.append("R69")
    diagnoses = [
        Diagnosis(rank=i, diagnosis="", icd10_code=code, explanation="")
        for i, code in enumerate(codes[:3], start=1)
    ]
    return DiagnoseResponse(diagnoses=diagnoses)
