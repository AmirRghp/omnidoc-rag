"""FastAPI routes for technical RAG querying, SSE streaming, and evaluation."""

import json
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_crag_workflow, verify_api_key
from src.api.errors import resolve_trace_id
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    QueryRequest,
    QueryResponse,
)
from src.common.logger import get_logger, set_trace_id
from src.common.schemas import CitationItem, RetrievedChunk
from src.graph.workflow import CRAGWorkflow

logger = get_logger(__name__)

router = APIRouter(tags=["Query & Streaming"])


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format a dictionary payload as a valid Server-Sent Event (SSE) block.

    Args:
        event: SSE event type name (e.g. 'step_start', 'token', 'done').
        data: Payload dictionary to be JSON-encoded.

    Returns:
        Formatted SSE text block ending with double newline.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def tokenize_generation_stream(text: str) -> list[str]:
    """Split generated response text into natural token slices preserving whitespace.

    Args:
        text: Complete generated answer text.

    Returns:
        Ordered list of token slices.
    """
    if not text:
        return []
    chunks = re.findall(r"\s*\S+", text)
    if not chunks and text:
        return [text]
    matched_len = sum(len(c) for c in chunks)
    if matched_len < len(text):
        chunks.append(text[matched_len:])
    return chunks


async def sse_query_generator(
    request: QueryRequest,
    workflow: CRAGWorkflow,
    trace_id: str,
) -> AsyncIterator[str]:
    """Asynchronous generator streaming LangGraph CRAG execution steps and tokens via SSE.

    Args:
        request: Validated user query request.
        workflow: CRAGWorkflow execution engine.
        trace_id: Correlation trace identifier.

    Yields:
        Formatted SSE string blocks for each step, token, citations, and completion.
    """
    start_time = time.perf_counter()
    tokens_used = 0

    try:
        # 1. Emit initial retrieval step
        yield format_sse(
            "step_start",
            {
                "event": "step_start",
                "step": "hybrid_retrieval",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        final_citations: list[CitationItem | dict[str, Any]] = []

        # 2. Iterate node transitions through LangGraph CRAG workflow
        async for event in workflow.astream(
            query=request.query,
            trace_id=trace_id,
            thread_id=request.session_id,
        ):
            # Case: Retrieval node finished
            if "retrieve" in event:
                node_data = event["retrieve"]
                reranked_chunks: list[RetrievedChunk] = node_data.get("reranked_chunks", [])
                top_score = max((float(c.score) for c in reranked_chunks), default=0.0)

                seen_sources: set[str] = set()
                sources: list[str] = []
                for chunk in reranked_chunks:
                    if chunk.source_file not in seen_sources:
                        seen_sources.add(chunk.source_file)
                        sources.append(chunk.source_file)

                yield format_sse(
                    "retrieval_complete",
                    {
                        "event": "retrieval_complete",
                        "count": len(reranked_chunks),
                        "top_score": round(top_score, 3),
                        "sources": sources,
                    },
                )

            # Case: CRAG document grading completed
            elif "grade_documents" in event:
                node_data = event["grade_documents"]
                crag_score = float(node_data.get("crag_score", 0.0))
                status = str(node_data.get("relevance_status", "incorrect"))
                yield format_sse(
                    "step",
                    {
                        "event": "step",
                        "step": "evaluate",
                        "status": status,
                        "score": round(crag_score, 3),
                    },
                )

            # Case: Ambiguous query rewritten
            elif "rewrite_query" in event:
                node_data = event["rewrite_query"]
                rewritten_query = str(node_data.get("rewritten_query", ""))
                yield format_sse(
                    "step",
                    {
                        "event": "step",
                        "step": "rewrite_query",
                        "query": rewritten_query,
                    },
                )

            # Case: Grounded answer generation completed
            elif "generate" in event:
                node_data = event["generate"]
                generation_text = str(node_data.get("generation", ""))
                final_citations = node_data.get("citations", [])

                token_slices = tokenize_generation_stream(generation_text)
                tokens_used = len(token_slices)
                for tok in token_slices:
                    yield format_sse("token", {"event": "token", "content": tok})

            # Case: Low-relevance refusal emitted
            elif "refuse" in event:
                node_data = event["refuse"]
                refusal_text = str(node_data.get("generation", ""))
                final_citations = []

                token_slices = tokenize_generation_stream(refusal_text)
                tokens_used = len(token_slices)
                for tok in token_slices:
                    yield format_sse("token", {"event": "token", "content": tok})

        # 3. Emit final citations list
        citations_payload: list[dict[str, Any]] = []
        for cit in final_citations:
            if hasattr(cit, "model_dump"):
                citations_payload.append(cit.model_dump())
            elif isinstance(cit, dict):
                citations_payload.append(cit)

        yield format_sse(
            "citations",
            {
                "event": "citations",
                "citations": citations_payload,
            },
        )

        # 4. Emit stream completion event
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        yield format_sse(
            "done",
            {
                "event": "done",
                "trace_id": trace_id,
                "latency_ms": elapsed_ms,
                "tokens_used": tokens_used,
            },
        )

    except Exception as exc:
        logger.error(
            "streaming_pipeline_error",
            trace_id=trace_id,
            error=str(exc),
            exc_info=True,
        )
        yield format_sse(
            "error",
            {
                "event": "error",
                "message": "An error occurred during query stream generation.",
                "detail": str(exc),
                "trace_id": trace_id,
            },
        )


@router.post(
    "/query/stream",
    summary="Stream technical query execution via Server-Sent Events",
    dependencies=[Depends(verify_api_key)],
    responses={
        200: {
            "description": "SSE stream delivering steps, tokens, and citations.",
            "content": {"text/event-stream": {}},
        }
    },
)
async def query_stream_endpoint(
    request: QueryRequest,
    req: Request,
    workflow: Annotated[CRAGWorkflow, Depends(get_crag_workflow)],
) -> StreamingResponse:
    """Stream LangGraph CRAG step transitions, tokens, and citations via Server-Sent Events."""
    trace_id = resolve_trace_id(req)
    set_trace_id(trace_id)

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Trace-ID": trace_id,
    }

    return StreamingResponse(
        sse_query_generator(request=request, workflow=workflow, trace_id=trace_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Synchronous grounded technical query execution",
    dependencies=[Depends(verify_api_key)],
)
async def query_sync_endpoint(
    request: QueryRequest,
    req: Request,
    workflow: Annotated[CRAGWorkflow, Depends(get_crag_workflow)],
) -> QueryResponse:
    """Execute complete CRAG workflow synchronously and return final grounded answer."""
    trace_id = resolve_trace_id(req)
    set_trace_id(trace_id)
    start_time = time.perf_counter()

    final_state = await workflow.ainvoke(
        query=request.query,
        trace_id=trace_id,
        thread_id=request.session_id,
    )
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    raw_status = final_state.get("relevance_status", "incorrect")
    status_str = str(raw_status).upper()
    if status_str not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
        status_str = "INCORRECT"

    citations = final_state.get("citations", [])
    citation_items: list[CitationItem] = [
        c if isinstance(c, CitationItem) else CitationItem.model_validate(c)
        for c in citations
    ]

    return QueryResponse(
        query=request.query,
        answer=str(final_state.get("generation", "")),
        citations=citation_items,
        reflection_status=status_str,  # type: ignore[arg-type]
        latency_ms=latency_ms,
        trace_id=trace_id,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit user feedback on a generated response",
    dependencies=[Depends(verify_api_key)],
)
async def feedback_endpoint(
    request: FeedbackRequest,
) -> FeedbackResponse:
    """Record user thumbs-up/down rating and comments for response evaluation."""
    logger.info(
        "user_feedback_received",
        trace_id=request.trace_id,
        rating=request.rating,
        score=request.score,
        comment=request.comment,
    )
    return FeedbackResponse(
        status="recorded",
        trace_id=request.trace_id,
        message="Feedback recorded successfully.",
    )
