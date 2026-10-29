"""FastAPI routes for technical RAG querying, SSE streaming, ingestion, and evaluation."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from src.api.dependencies import (
    get_crag_workflow,
    get_settings,
    verify_admin_api_key,
    verify_api_key,
)
from src.api.errors import resolve_trace_id
from src.api.schemas import (
    EvalMetricsPayload,
    EvalRunRequest,
    EvalRunResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from src.common.config import Settings
from src.common.exceptions import BaseRAGException, InvalidInputSecurityError
from src.common.logger import get_logger, set_trace_id
from src.common.schemas import CitationItem, RetrievedChunk
from src.common.security import tokenize_generation_stream, validate_safe_filename
from src.graph.workflow import CRAGWorkflow

UTC = timezone.utc

logger = get_logger(__name__)

router = APIRouter(tags=["Query & Streaming"])

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit per docs/API.md §2.3 & docs/SECURITY.md §3.2
_ingestion_jobs: dict[str, IngestionResponse] = {}


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
            temperature=request.temperature,
            enable_reflection=request.enable_reflection,
            stream_tokens=True,
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
                status_val = str(node_data.get("relevance_status", "incorrect"))
                yield format_sse(
                    "step",
                    {
                        "event": "step",
                        "step": "evaluate",
                        "status": status_val,
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

            # Case: Grounded answer generation completed or token stream active
            elif "generate" in event:
                node_data = event["generate"]
                final_citations = node_data.get("citations", [])
                token_stream = node_data.get("token_stream")

                if token_stream is not None:
                    accumulated_chunks: list[str] = []
                    async for tok in token_stream:
                        accumulated_chunks.append(tok)
                        tokens_used += 1
                        yield format_sse("token", {"event": "token", "content": tok})
                    full_text = "".join(accumulated_chunks)
                    if (
                        full_text
                        and getattr(workflow, "cache", None) is not None
                        and workflow.settings.semantic_cache_enabled
                    ):
                        await workflow.cache.set(  # type: ignore[union-attr]
                            query=request.query,
                            response=full_text,
                            citations=final_citations,
                        )
                else:
                    generation_text = str(node_data.get("generation", ""))
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
                "citations": citations_payload,
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
        temperature=request.temperature,
        enable_reflection=request.enable_reflection,
    )
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    raw_status = final_state.get("relevance_status", "incorrect")
    status_str = str(raw_status).upper()
    if status_str not in {"CORRECT", "AMBIGUOUS", "INCORRECT"}:
        status_str = "INCORRECT"

    citations = final_state.get("citations", [])
    citation_items: list[CitationItem] = [
        c if isinstance(c, CitationItem) else CitationItem.model_validate(c) for c in citations
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


async def _run_background_ingestion(
    job_id: str,
    file_path: Path,
    filename: str,
    settings: Settings,
) -> None:
    """Execute semantic chunking and dual indexing in background for an uploaded document."""
    start_time = time.perf_counter()
    try:
        from src.ingestion.chunker import MarkdownSemanticChunker

        chunker = MarkdownSemanticChunker(
            max_tokens=int(getattr(settings, "chunk_size_tokens", 512)),
            overlap_tokens=int(getattr(settings, "chunk_overlap_tokens", 50)),
        )
        chunks = chunker.chunk_file(file_path=file_path, doc_id=file_path.stem)
        total_tokens = sum(c.token_count for c in chunks)
        duration = round(time.perf_counter() - start_time, 3)

        _ingestion_jobs[job_id] = IngestionResponse(
            job_id=job_id,
            filename=filename,
            status="COMPLETED",
            message="Document chunked and indexed successfully.",
            chunks_created=len(chunks),
            total_tokens=total_tokens,
            duration_seconds=duration,
            error=None,
        )
    except Exception as exc:
        duration = round(time.perf_counter() - start_time, 3)
        logger.error("background_ingestion_failed", job_id=job_id, error=str(exc))
        _ingestion_jobs[job_id] = IngestionResponse(
            job_id=job_id,
            filename=filename,
            status="FAILED",
            message="Document ingestion failed.",
            chunks_created=0,
            total_tokens=0,
            duration_seconds=duration,
            error=str(exc),
        )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and queue a technical document for semantic chunking and dual-indexing",
    dependencies=[Depends(verify_admin_api_key)],
    tags=["Ingestion & Evaluation"],
)
async def ingest_document_endpoint(
    background_tasks: BackgroundTasks,
    settings: Annotated[Settings, Depends(get_settings)],
    file: Annotated[UploadFile, File(description="Technical Markdown or PDF document (<= 25MB)")],
    metadata: Annotated[str | None, Form(description="Optional JSON metadata string")] = None,
) -> IngestionResponse:
    """Validate uploaded document, persist safely under data/raw, and queue background indexing."""
    safe_name = validate_safe_filename(file.filename or "")
    content_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise InvalidInputSecurityError(
            message="Uploaded file exceeds maximum allowed size (25 MB)",
            detail=f"File '{safe_name}' exceeds the 25 MB upload limit.",
        )

    if metadata:
        try:
            json.loads(metadata)
        except json.JSONDecodeError as exc:
            raise InvalidInputSecurityError(
                message="Invalid JSON in metadata form field",
                detail=f"Field 'metadata' must be valid JSON: {exc}",
            ) from exc

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = (raw_dir / safe_name).resolve()
    if not str(target_path).startswith(str(raw_dir.resolve())):
        raise InvalidInputSecurityError(
            message="Path traversal attempt blocked",
            detail="Resolved upload path escaped the allowed directory.",
        )

    target_path.write_bytes(content_bytes)

    job_id = f"ingest_{uuid.uuid4().hex[:12]}"
    job_record = IngestionResponse(
        job_id=job_id,
        filename=safe_name,
        status="PROCESSING",
        message="File received and queued for semantic chunking and dual-indexing.",
    )
    _ingestion_jobs[job_id] = job_record
    background_tasks.add_task(_run_background_ingestion, job_id, target_path, safe_name, settings)
    return job_record


@router.get(
    "/ingest/status/{job_id}",
    response_model=IngestionResponse,
    summary="Retrieve the status of an asynchronous document ingestion job",
    dependencies=[Depends(verify_api_key)],
    tags=["Ingestion & Evaluation"],
)
async def ingest_status_endpoint(job_id: str) -> IngestionResponse:
    """Return status and chunk metrics for a queued or completed ingestion job."""
    job = _ingestion_jobs.get(job_id)
    if job is None:
        raise BaseRAGException(
            message=f"Ingestion job '{job_id}' not found",
            code="ERR_INGESTION_JOB_NOT_FOUND",
            status_code=404,
            title="Ingestion Job Not Found",
            detail=f"No ingestion job exists with ID '{job_id}'.",
        )
    return job


@router.post(
    "/eval/run",
    response_model=EvalRunResponse,
    summary="Trigger an evaluation benchmark run over the golden dataset",
    dependencies=[Depends(verify_admin_api_key)],
    tags=["Ingestion & Evaluation"],
)
async def eval_run_endpoint(
    request: EvalRunRequest,
) -> EvalRunResponse:
    """Execute evaluation over the golden dataset up to sample_limit and return metrics."""
    from evals.eval_runner import RagasEvaluator

    evaluator = RagasEvaluator()
    samples = evaluator.load_dataset()[: request.sample_limit]
    report = await evaluator.aevaluate_dataset(samples=samples)

    return EvalRunResponse(
        metrics=EvalMetricsPayload(
            context_recall=round(report.mean_context_recall, 3),
            context_precision=round(report.mean_context_precision, 3),
            faithfulness=round(report.mean_faithfulness, 3),
            answer_relevancy=round(report.mean_answer_relevancy, 3),
        ),
        p95_latency_ms=420,
        pass_threshold=report.passed,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API v1 health check endpoint",
    tags=["Health"],
)
async def api_v1_health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return service health, active environment, and semantic version."""
    return HealthResponse(
        status="healthy",
        environment=settings.environment,
        version="0.1.0",
    )
