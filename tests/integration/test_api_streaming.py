"""Integration tests for FastAPI SSE streaming endpoint, headers, and contracts (Task 8.3)."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from src.api.dependencies import get_crag_workflow
from src.common.config import Settings
from src.common.schemas import CitationItem, RetrievedChunk
from src.graph.state import RAGState
from src.graph.workflow import CRAGWorkflow
from src.main import create_app

VALID_API_KEY = "dev_insecure_local_key_replace_in_prod"
AUTH_HEADERS = {"Authorization": f"Bearer {VALID_API_KEY}"}


def parse_sse_events(raw_body: str) -> list[dict[str, Any]]:
    """Parse raw SSE text stream body into a list of event objects.

    Each event dict contains 'event' (event type) and 'data' (parsed JSON dictionary).
    """
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    current_data_lines: list[str] = []

    for line in raw_body.splitlines():
        line = line.strip()
        if not line:
            if current_event or current_data_lines:
                data_str = "\n".join(current_data_lines)
                parsed_data = json.loads(data_str) if data_str else {}
                events.append({"event": current_event, "data": parsed_data})
                current_event = None
                current_data_lines = []
            continue

        if line.startswith("event:"):
            current_event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            current_data_lines.append(line.removeprefix("data:").strip())

    if current_event or current_data_lines:
        data_str = "\n".join(current_data_lines)
        parsed_data = json.loads(data_str) if data_str else {}
        events.append({"event": current_event, "data": parsed_data})

    return events


@pytest.fixture
def test_app() -> FastAPI:
    """Create test application instance with testing configuration."""
    settings = Settings(
        ENVIRONMENT="testing",
        API_KEY=VALID_API_KEY,
    )
    return create_app(settings=settings)


@pytest.mark.asyncio
class TestAPIStreamingEndpoint:
    """Integration test suite for POST /api/v1/query/stream and SSE protocol."""

    async def test_query_stream_happy_path(self, test_app: FastAPI) -> None:
        """Standard query yields step_start, retrieval_complete, step, token, citations, done."""
        sample_chunk = RetrievedChunk(
            chunk_id="chk_docker_1",
            source_file="manuals/docker_run.md",
            content="Use the -P or --publish-all flag to publish all exposed ports.",
            score=0.892,
            metadata={"section_heading": "Exposing and Publishing Ports"},
        )
        sample_citation = CitationItem(
            source_file="manuals/docker_run.md",
            section_heading="Exposing and Publishing Ports",
            relevance_score=0.892,
            snippet="Use the -P or --publish-all flag to publish all exposed ports.",
        )

        mock_workflow = AsyncMock(spec=CRAGWorkflow)

        async def fake_astream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {
                "retrieve": {
                    "retrieved_chunks": [sample_chunk],
                    "reranked_chunks": [sample_chunk],
                }
            }
            yield {
                "grade_documents": {
                    "crag_score": 0.892,
                    "relevance_status": "correct",
                }
            }
            yield {
                "generate": {
                    "generation": "برای نگاشت پورت‌ها از فلگ -P استفاده کنید.",
                    "citations": [sample_citation],
                }
            }

        mock_workflow.astream = fake_astream
        test_app.dependency_overrides[get_crag_workflow] = lambda: mock_workflow

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/stream",
                json={
                    "query": "چگونه می‌توان پورت‌های کانتینر داکر را به صورت داینامیک نگاشت کرد؟",
                    "session_id": "sess_test_123",
                    "temperature": 0.2,
                },
                headers={**AUTH_HEADERS, "Accept": "text/event-stream"},
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            assert "no-cache" in response.headers["cache-control"]
            assert "X-Trace-ID" in response.headers

            events = parse_sse_events(response.text)
            event_types = [e["event"] for e in events]

            assert "step_start" in event_types
            assert "retrieval_complete" in event_types
            assert "step" in event_types
            assert "token" in event_types
            assert "citations" in event_types
            assert "done" in event_types

            # Verify step_start
            step_start_event = next(e for e in events if e["event"] == "step_start")
            assert step_start_event["data"]["step"] == "hybrid_retrieval"

            # Verify retrieval_complete
            retrieval_event = next(e for e in events if e["event"] == "retrieval_complete")
            assert retrieval_event["data"]["count"] == 1
            assert retrieval_event["data"]["top_score"] == 0.892
            assert "manuals/docker_run.md" in retrieval_event["data"]["sources"]

            # Verify evaluation step
            eval_event = next(e for e in events if e["event"] == "step")
            assert eval_event["data"]["step"] == "evaluate"
            assert eval_event["data"]["status"] == "correct"

            # Verify streamed tokens reconstruct the answer
            token_events = [e for e in events if e["event"] == "token"]
            reconstructed_answer = "".join(e["data"]["content"] for e in token_events)
            assert reconstructed_answer == "برای نگاشت پورت‌ها از فلگ -P استفاده کنید."

            # Verify citations payload
            citations_event = next(e for e in events if e["event"] == "citations")
            assert len(citations_event["data"]["citations"]) == 1
            assert citations_event["data"]["citations"][0]["source_file"] == "manuals/docker_run.md"

            # Verify completion event
            done_event = next(e for e in events if e["event"] == "done")
            assert done_event["data"]["tokens_used"] > 0
            assert done_event["data"]["latency_ms"] >= 0
            assert len(done_event["data"]["trace_id"]) > 0

    async def test_query_stream_refusal_path(self, test_app: FastAPI) -> None:
        """Low-relevance query streams polite refusal without citations."""
        mock_workflow = AsyncMock(spec=CRAGWorkflow)

        async def fake_astream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {
                "retrieve": {
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                }
            }
            yield {
                "grade_documents": {
                    "crag_score": 0.12,
                    "relevance_status": "incorrect",
                }
            }
            yield {
                "refuse": {
                    "generation": "اطلاعات مربوط به این سوال در مستندات فنی یافت نشد.",
                    "citations": [],
                    "refusal_reason": "insufficient_context",
                }
            }

        mock_workflow.astream = fake_astream
        test_app.dependency_overrides[get_crag_workflow] = lambda: mock_workflow

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/stream",
                json={"query": "طرز تهیه غذای ایتالیایی"},
                headers=AUTH_HEADERS,
            )

            assert response.status_code == 200
            events = parse_sse_events(response.text)
            event_types = [e["event"] for e in events]

            assert "step_start" in event_types
            assert "retrieval_complete" in event_types
            assert "step" in event_types
            assert "token" in event_types
            assert "citations" in event_types
            assert "done" in event_types

            # Citations must be empty for refusal
            citations_event = next(e for e in events if e["event"] == "citations")
            assert citations_event["data"]["citations"] == []

            # Reconstructed token text must contain refusal wording
            token_events = [e for e in events if e["event"] == "token"]
            refusal_text = "".join(e["data"]["content"] for e in token_events)
            assert "مستندات" in refusal_text or "یافت نشد" in refusal_text

    async def test_query_stream_ambiguous_rewrite_step(self, test_app: FastAPI) -> None:
        """Ambiguous query streams intermediate rewrite_query step."""
        mock_workflow = AsyncMock(spec=CRAGWorkflow)

        async def fake_astream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {
                "retrieve": {
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                }
            }
            yield {
                "grade_documents": {
                    "crag_score": 0.45,
                    "relevance_status": "ambiguous",
                }
            }
            yield {
                "rewrite_query": {
                    "rewritten_query": "Docker bridge network port mapping",
                }
            }
            yield {
                "retrieve": {
                    "retrieved_chunks": [],
                    "reranked_chunks": [],
                }
            }
            yield {
                "grade_documents": {
                    "crag_score": 0.88,
                    "relevance_status": "correct",
                }
            }
            yield {
                "generate": {
                    "generation": "Rewritten response answer.",
                    "citations": [],
                }
            }

        mock_workflow.astream = fake_astream
        test_app.dependency_overrides[get_crag_workflow] = lambda: mock_workflow

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/stream",
                json={"query": "داکر پورت"},
                headers=AUTH_HEADERS,
            )

            assert response.status_code == 200
            events = parse_sse_events(response.text)

            rewrite_events = [
                e
                for e in events
                if e["event"] == "step" and e["data"].get("step") == "rewrite_query"
            ]
            assert len(rewrite_events) == 1
            assert rewrite_events[0]["data"]["query"] == "Docker bridge network port mapping"

    async def test_query_stream_validation_error_returns_rfc7807(self, test_app: FastAPI) -> None:
        """Invalid empty query returns RFC 7807 problem details with 400 or 422 status."""
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/stream",
                json={"query": "   "},
                headers=AUTH_HEADERS,
            )

            assert response.status_code in {400, 422}
            data = response.json()
            assert "type" in data
            assert "title" in data
            assert "status" in data
            assert "trace_id" in data
            assert "instance" in data

    async def test_query_stream_unauthorized_rejection(self, test_app: FastAPI) -> None:
        """Missing or invalid Authorization header returns 401 Authentication Error."""
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Missing header
            res_missing = await client.post(
                "/api/v1/query/stream",
                json={"query": "test query"},
            )
            assert res_missing.status_code == 401
            assert res_missing.json()["code"] == "ERR_AUTHENTICATION_FAILED"

            # Malformed header
            res_malformed = await client.post(
                "/api/v1/query/stream",
                json={"query": "test query"},
                headers={"Authorization": "Basic 12345"},
            )
            assert res_malformed.status_code == 401

            # Invalid key
            res_invalid = await client.post(
                "/api/v1/query/stream",
                json={"query": "test query"},
                headers={"Authorization": "Bearer totally_wrong_secret_key"},
            )
            assert res_invalid.status_code == 401

    async def test_query_stream_pipeline_exception_yields_error_event(
        self, test_app: FastAPI
    ) -> None:
        """Unexpected exception in workflow generator yields SSE error frame without hanging."""
        mock_workflow = AsyncMock(spec=CRAGWorkflow)

        async def error_astream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            yield {"retrieve": {"reranked_chunks": []}}
            raise RuntimeError("Database connection suddenly dropped")

        mock_workflow.astream = error_astream
        test_app.dependency_overrides[get_crag_workflow] = lambda: mock_workflow

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query/stream",
                json={"query": "Docker swarm setup"},
                headers=AUTH_HEADERS,
            )

            assert response.status_code == 200
            events = parse_sse_events(response.text)
            event_types = [e["event"] for e in events]

            assert "error" in event_types
            error_event = next(e for e in events if e["event"] == "error")
            assert "Database connection suddenly dropped" in error_event["data"]["detail"]
            assert "trace_id" in error_event["data"]

    async def test_sync_query_endpoint(self, test_app: FastAPI) -> None:
        """Synchronous /query endpoint executes ainvoke and returns QueryResponse."""
        sample_chunk = RetrievedChunk(
            chunk_id="chk_sync_1",
            source_file="manuals/linux_cgroups.md",
            content="Control groups allocate CPU limits.",
            score=0.92,
            metadata={"section_heading": "Cgroups"},
        )
        final_state: RAGState = {
            "query": "How do Linux cgroups work?",
            "rewritten_query": None,
            "language": "en",
            "retrieved_chunks": [sample_chunk],
            "reranked_chunks": [sample_chunk],
            "crag_score": 0.92,
            "relevance_status": "correct",
            "transformation_count": 0,
            "generation": "Linux cgroups provide resource isolation for processes.",
            "citations": [
                CitationItem(
                    source_file="manuals/linux_cgroups.md",
                    section_heading="Cgroups",
                    relevance_score=0.92,
                    snippet="Control groups allocate CPU limits.",
                )
            ],
            "refusal_reason": None,
            "error": None,
            "trace_id": "tr_sync_test_42",
        }

        mock_workflow = AsyncMock(spec=CRAGWorkflow)
        mock_workflow.ainvoke = AsyncMock(return_value=final_state)
        test_app.dependency_overrides[get_crag_workflow] = lambda: mock_workflow

        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "How do Linux cgroups work?"},
                headers=AUTH_HEADERS,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["query"] == "How do Linux cgroups work?"
            assert data["answer"] == "Linux cgroups provide resource isolation for processes."
            assert len(data["citations"]) == 1
            assert data["reflection_status"] == "CORRECT"
            assert data["latency_ms"] >= 0

    async def test_feedback_endpoint(self, test_app: FastAPI) -> None:
        """User feedback endpoint records rating and returns confirmation."""
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/feedback",
                json={
                    "trace_id": "tr_feedback_123",
                    "rating": "up",
                    "score": 5,
                    "comment": "Great accurate answer!",
                },
                headers=AUTH_HEADERS,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "recorded"
            assert data["trace_id"] == "tr_feedback_123"

    async def test_health_check_endpoint(self, test_app: FastAPI) -> None:
        """GET /health returns healthy status."""
        transport = httpx.ASGITransport(app=test_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
