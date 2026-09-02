"""FastAPI service in front of the retrieval pipeline.

Why a service at all when Lambda already calls the library directly: the
Lambda is one caller. An agent-assist panel that suggests answers to a human
agent, a batch job that replays yesterday's transcripts to measure whether a
prompt change would have helped, and a local debugging client are three more,
and none of them should have to run inside Lex to ask a question.

It is also the honest place for the observability the JD asks about. Every
response carries its own latency, cost and token usage, so a caller can see
what a turn cost without reading a dashboard - and `/metrics` aggregates
containment across contacts.

    uvicorn src.service:app --reload
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import metrics, prompts, rag
from .schemas import AskRequest, AskResponse, Citation, HealthResponse, TokenUsage

logger = logging.getLogger("connect-rag")

app = FastAPI(
    title="Connect RAG service",
    version="1.0.0",
    description="Grounded answering with abstention enforced in code.",
)

# Swapped in tests and at startup. Module-level rather than a global singleton
# built at import, so importing this module never opens a connection.
STORE = None
GENERATE = None
CONTACTS: dict[str, metrics.Contact] = {}


@app.middleware("http")
async def correlate_and_time(request: Request, call_next):
    """One request id through logs and back to the caller.

    Without it, a slow turn reported by a customer cannot be found in the logs
    - which is the moment observability either exists or does not.
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("unhandled", extra={"request_id": request_id})
        return JSONResponse(status_code=500,
                            content={"detail": "internal error",
                                     "request_id": request_id})
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-latency-ms"] = f"{elapsed_ms:.1f}"
    logger.info("request", extra={"request_id": request_id,
                                  "path": request.url.path,
                                  "status": response.status_code,
                                  "latency_ms": round(elapsed_ms, 1)})
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        chunks_indexed=len(getattr(STORE, "docs", []) or []),
        min_passages=rag.MIN_PASSAGES,
        min_relevance=rag.MIN_RELEVANCE,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """Answer, or decline and say so.

    The response model enforces that an ungrounded answer sets escalate=true,
    so this handler cannot ship the failure mode where a customer is told "I
    don't know" and then left in a loop with the bot.
    """
    with metrics.Stopwatch() as timer:
        result = await rag.answer_async(payload.question, STORE, GENERATE)

    _, packed = prompts.build_prompt(payload.question, result.passages)

    turn = metrics.Turn(
        intent="AskQuestion",
        grounded=result.grounded,
        escalated=result.should_escalate,
        latency_ms=timer.elapsed_ms,
    )
    CONTACTS.setdefault(payload.contact_id,
                        metrics.Contact(payload.contact_id)).record(turn)

    logger.info("answered", extra={
        "contact_id": payload.contact_id,
        "grounded": result.grounded,
        "reason": result.reason,
        "passages": len(result.passages),
        "context_tokens": packed.estimated_tokens,
        "dropped_passages": packed.dropped,
        "latency_ms": round(timer.elapsed_ms, 1),
    })

    return AskResponse(
        answer=result.text,
        grounded=result.grounded,
        escalate=result.should_escalate,
        reason=result.reason,
        citations=[Citation(source=p.source, score=round(p.score, 3),
                            excerpt=p.text[:280]) for p in result.passages],
        latency_ms=round(timer.elapsed_ms, 2),
        cost_usd=round(turn.cost_usd, 6),
        tokens=TokenUsage(prompt=packed.estimated_tokens,
                          completion=prompts.estimate_tokens(result.text),
                          context_passages=len(packed.used)),
    )


@app.get("/metrics")
async def service_metrics() -> dict:
    """Containment, escalation, cost and p95 across everything seen so far."""
    return metrics.summary(list(CONTACTS.values()))
