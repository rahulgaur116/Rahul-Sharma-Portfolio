"""CapexBench API - parametric data center capex benchmarking.

Run from the project root:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /api/health      - service + model status
    GET  /api/dimensions  - queryable dimensions and values
    POST /api/benchmark   - benchmark stats over comparable projects
    POST /api/estimate    - parametric P10/P50/P90 capex estimate
    POST /api/ask         - AI natural-language query layer (needs API key)
    GET  /                - web app
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from capexbench import benchmark as bench
from capexbench.dataset import load
from capexbench.predict import estimate_capex

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"

app = FastAPI(title="CapexBench", version="0.1.0",
              description="Parametric benchmarking for data center capex")

_df = load()

# In-memory AI chat sessions (demo-grade; swap for a store in production)
_sessions: dict[str, list] = {}
_MAX_SESSIONS = 500


class BenchmarkRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)


class EstimateRequest(BaseModel):
    facility_type: str | None = None
    market: str | None = None
    delivery_year: int | None = None
    it_load_mw: float | None = None
    kw_per_rack: float | None = None
    cooling: str | None = None
    redundancy: str | None = None
    build_type: str | None = None


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "n_projects": int(len(_df)),
        "ai_query_layer": bool(os.environ.get("ANTHROPIC_API_KEY")
                               or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
    }


@app.get("/api/dimensions")
def dimensions() -> dict:
    return bench.dimensions(_df)


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest) -> dict:
    try:
        return bench.query_benchmarks(_df, req.filters)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/estimate")
def estimate(req: EstimateRequest) -> dict:
    try:
        return estimate_capex(req.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/api/ask")
def ask_ai(req: AskRequest) -> dict:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise HTTPException(
            status_code=503,
            detail="AI query layer is not configured - set ANTHROPIC_API_KEY.",
        )
    from capexbench.ai_query import ask  # deferred: requires anthropic credentials

    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])
    try:
        result = ask(req.question, history=history)
    except Exception as e:  # surface API/auth errors cleanly
        raise HTTPException(status_code=502, detail=f"AI query failed: {e}")

    if len(_sessions) >= _MAX_SESSIONS and session_id not in _sessions:
        _sessions.pop(next(iter(_sessions)))
    _sessions[session_id] = result["history"]
    return {
        "answer": result["answer"],
        "tool_calls": result["tool_calls"],
        "session_id": session_id,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
