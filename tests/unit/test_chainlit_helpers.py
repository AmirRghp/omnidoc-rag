"""Unit tests for Chainlit chat workflow, steps, citations drawer, and feedback (Task 9.2)."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app
from src.common.schemas import CitationItem, RetrievedChunk


class FakeStep:
    """Test double for chainlit.Step context manager."""

    instances: list["FakeStep"] = []

    def __init__(self, name: str = "", type: str = "undefined", **kwargs: Any) -> None:
        self.name = name
        self.type = type
        self.input: str = ""
        self.output: str = ""
        self.kwargs = kwargs
        FakeStep.instances.append(self)

    async def __aenter__(self) -> "FakeStep":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    async def send(self) -> "FakeStep":
        return self

    async def update(self) -> bool:
        return True


class FakeMessage:
    """Test double for chainlit.Message."""

    instances: list["FakeMessage"] = []

    def __init__(
        self,
        content: str = "",
        author: str = "Assistant",
        elements: list[Any] | None = None,
        actions: list[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.content = content
        self.author = author
        self.elements = list(elements) if elements else []
        self.actions = list(actions) if actions else []
        self.streamed_tokens: list[str] = []
        self.sent = False
        self.updated = False
        self.kwargs = kwargs
        FakeMessage.instances.append(self)

    async def stream_token(self, token: str) -> None:
        self.streamed_tokens.append(token)
        self.content += token

    async def send(self) -> "FakeMessage":
        self.sent = True
        return self

    async def update(self) -> bool:
        self.updated = True
        return True


class FakeText:
    """Test double for chainlit.Text side-drawer element."""

    def __init__(
        self,
        name: str,
        content: str,
        display: str = "side",
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.content = content
        self.display = display
        self.kwargs = kwargs


class FakeAction:
    """Test double for chainlit.Action feedback button."""

    def __init__(
        self,
        name: str,
        payload: dict[str, Any],
        label: str = "",
        tooltip: str = "",
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.payload = payload
        self.label = label
        self.tooltip = tooltip
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    """Clear recorded FakeStep and FakeMessage instances before each test."""
    FakeStep.instances.clear()
    FakeMessage.instances.clear()


def _sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=0,
            source_file="docker_networking_bridge.md",
            content="Bridge networks apply to containers running on the same Docker daemon host.",
            score=0.912,
            metadata={"section_heading": "Bridge Network Overview"},
        ),
        RetrievedChunk(
            chunk_id=4,
            source_file="docker_compose_file_spec.md",
            content="Use network_mode: bridge or custom bridge networks in compose.yaml.",
            score=0.834,
            metadata={"section_heading": "Network Mode Configuration"},
        ),
    ]


def _sample_citations() -> list[CitationItem]:
    return [
        CitationItem(
            source_file="docker_networking_bridge.md",
            section_heading="Bridge Network Overview",
            relevance_score=0.912,
            snippet="Bridge networks apply to containers running on the same Docker daemon host.",
        ),
        CitationItem(
            source_file="docker_compose_file_spec.md",
            section_heading="Network Mode Configuration",
            relevance_score=0.834,
            snippet="Use network_mode: bridge or custom bridge networks in compose.yaml.",
        ),
    ]


def test_step_constants_match_plan_spec() -> None:
    """Verify the three required collapsible step titles match Task 9.2 specification."""
    assert app.STEP_RETRIEVAL_NAME == "🔍 Retrieval (Dense + Sparse)"
    assert app.STEP_RERANK_NAME == "⚖️ Fusion & Reranking"
    assert app.STEP_CRAG_NAME == "🎯 Quality Gate (CRAG)"


def test_format_retrieval_step_output() -> None:
    """Verify retrieval step formatting includes candidate count, sources, and active query."""
    chunks = _sample_chunks()
    output = app.format_retrieval_step_output(chunks, query="نحوه تنظیم شبکه bridge در داکر")

    assert "2" in output
    assert "نحوه تنظیم شبکه bridge در داکر" in output
    assert "docker_networking_bridge.md" in output

    empty_output = app.format_retrieval_step_output([], query="سوال بدون نتیجه")
    assert "0" in empty_output


def test_format_rerank_step_output() -> None:
    """Verify rerank step formatting lists ranked candidates with chunk IDs and scores."""
    chunks = _sample_chunks()
    output = app.format_rerank_step_output(chunks)

    assert "docker_networking_bridge.md" in output
    assert "chunk-0" in output
    assert "0.912" in output
    assert "docker_compose_file_spec.md" in output
    assert "chunk-4" in output
    assert "0.834" in output

    empty_output = app.format_rerank_step_output([])
    assert "No candidates" in empty_output or "هیچ" in empty_output


def test_format_crag_step_output_states() -> None:
    """Verify CRAG step output formats CORRECT, AMBIGUOUS (with rewrite), and INCORRECT."""
    correct_out = app.format_crag_step_output(status="correct", score=0.912)
    assert "CORRECT" in correct_out
    assert "0.912" in correct_out

    ambiguous_out = app.format_crag_step_output(
        status="ambiguous",
        score=0.48,
        rewritten_query="Docker nginx reverse proxy 502 bad gateway",
    )
    assert "AMBIGUOUS" in ambiguous_out
    assert app.REWRITE_ALERT_TEXT in ambiguous_out
    assert "Docker nginx reverse proxy 502 bad gateway" in ambiguous_out

    incorrect_out = app.format_crag_step_output(status="incorrect", score=0.15)
    assert "INCORRECT" in incorrect_out
    assert "0.150" in incorrect_out


def test_build_citation_elements_and_pills() -> None:
    """Verify citation pills and side-drawer cl.Text elements include source file and chunk ID."""
    citations = _sample_citations()
    chunks = _sample_chunks()

    with patch("app.cl.Text", FakeText):
        elements = app.build_citation_elements(citations, chunks=chunks)

    assert len(elements) == 2
    first_el = elements[0]
    assert first_el.display == "side"
    assert "docker_networking_bridge.md" in first_el.name
    assert "chunk-0" in first_el.name
    assert isinstance(first_el.content, str)
    assert "Bridge Network Overview" in first_el.content
    assert "0.912" in first_el.content
    assert "Bridge networks apply" in first_el.content

    pills_md = app.format_citation_pills_markdown(citations, chunks=chunks)
    assert "citation-pill" in pills_md
    assert first_el.name in pills_md
    assert elements[1].name in pills_md

    assert app.format_citation_pills_markdown([]) == ""


def test_build_feedback_actions() -> None:
    """Verify feedback action buttons include trace_id and positive/negative ratings."""
    with patch("app.cl.Action", FakeAction):
        actions = app.build_feedback_actions(trace_id="trace-12345")

    assert len(actions) == 2
    names = {a.name for a in actions}
    assert names == {"feedback_positive", "feedback_negative"}
    for action in actions:
        assert action.payload["trace_id"] == "trace-12345"
        assert action.payload["rating"] in {"positive", "negative"}


@pytest.mark.asyncio
async def test_on_message_happy_path_streams_steps_tokens_and_citations() -> None:
    """Verify on_message executes CRAG stream, renders 3 steps, tokens, and citations."""
    chunks = _sample_chunks()
    citations = _sample_citations()

    async def fake_astream(
        query: str,
        trace_id: str | None = None,
        language: str | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"retrieve": {"retrieved_chunks": chunks, "reranked_chunks": chunks}}
        yield {"grade_documents": {"crag_score": 0.912, "relevance_status": "correct"}}
        yield {
            "generate": {
                "generation": (
                    "برای تنظیم شبکه bridge در داکر از گزینه `--network bridge` استفاده کنید."
                ),
                "citations": citations,
                "relevance_status": "correct",
            }
        }

    mock_workflow = MagicMock()
    mock_workflow.astream = fake_astream

    incoming_msg = FakeMessage(content="نحوه تنظیم شبکه bridge در داکر چیست؟")

    with (
        patch("app.cl.Step", FakeStep),
        patch("app.cl.Message", FakeMessage),
        patch("app.cl.Text", FakeText),
        patch("app.cl.Action", FakeAction),
        patch("app.get_crag_workflow", return_value=mock_workflow),
    ):
        await app.on_message(incoming_msg)

    step_names = [s.name for s in FakeStep.instances]
    assert app.STEP_RETRIEVAL_NAME in step_names
    assert app.STEP_RERANK_NAME in step_names
    assert app.STEP_CRAG_NAME in step_names

    # Outgoing assistant message is the second FakeMessage (after incoming_msg)
    assistant_msgs = [m for m in FakeMessage.instances if m is not incoming_msg]
    assert len(assistant_msgs) == 1
    reply = assistant_msgs[0]
    assert len(reply.streamed_tokens) > 0
    assert "--network bridge" in reply.content
    assert "citation-pill" in reply.content
    assert len(reply.elements) == 2
    assert len(reply.actions) == 2


@pytest.mark.asyncio
async def test_on_message_ambiguous_rewrite_flow() -> None:
    """Verify on_message displays the Persian query rewrite alert when rewrite_query triggers."""
    chunks = _sample_chunks()
    citations = _sample_citations()

    async def fake_astream(
        query: str,
        trace_id: str | None = None,
        language: str | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"retrieve": {"retrieved_chunks": chunks[:1], "reranked_chunks": chunks[:1]}}
        yield {"grade_documents": {"crag_score": 0.45, "relevance_status": "ambiguous"}}
        yield {
            "rewrite_query": {
                "rewritten_query": "Docker Nginx reverse proxy 502 Bad Gateway",
                "transformation_count": 1,
            }
        }
        yield {"retrieve": {"retrieved_chunks": chunks, "reranked_chunks": chunks}}
        yield {"grade_documents": {"crag_score": 0.88, "relevance_status": "correct"}}
        yield {
            "generate": {
                "generation": (
                    "خطای 502 معمولاً به دلیل عدم ارتباط Nginx با کانتینر upstream رخ می‌دهد."
                ),
                "citations": citations,
                "relevance_status": "correct",
            }
        }

    mock_workflow = MagicMock()
    mock_workflow.astream = fake_astream

    incoming_msg = FakeMessage(content="مشکل ارور ۵۰۲")

    with (
        patch("app.cl.Step", FakeStep),
        patch("app.cl.Message", FakeMessage),
        patch("app.cl.Text", FakeText),
        patch("app.cl.Action", FakeAction),
        patch("app.get_crag_workflow", return_value=mock_workflow),
    ):
        await app.on_message(incoming_msg)

    assistant_msgs = [m for m in FakeMessage.instances if m is not incoming_msg]
    assert len(assistant_msgs) == 1
    reply = assistant_msgs[0]
    assert app.REWRITE_ALERT_TEXT in reply.content
    assert "502" in reply.content


@pytest.mark.asyncio
async def test_on_message_refusal_flow_and_invalid_input() -> None:
    """Verify refusal flow emits zero citations and invalid input is rejected gracefully."""

    async def fake_refuse_stream(
        query: str,
        trace_id: str | None = None,
        language: str | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"retrieve": {"retrieved_chunks": [], "reranked_chunks": []}}
        yield {"grade_documents": {"crag_score": 0.0, "relevance_status": "incorrect"}}
        yield {
            "refuse": {
                "generation": "پاسخ این سوال در مستندات بارگذاری‌شده موجود نیست.",
                "citations": [],
                "relevance_status": "incorrect",
            }
        }

    mock_workflow = MagicMock()
    mock_workflow.astream = fake_refuse_stream

    incoming_msg = FakeMessage(content="قیمت سهام اپل چقدر است؟")

    with (
        patch("app.cl.Step", FakeStep),
        patch("app.cl.Message", FakeMessage),
        patch("app.cl.Text", FakeText),
        patch("app.cl.Action", FakeAction),
        patch("app.get_crag_workflow", return_value=mock_workflow),
    ):
        await app.on_message(incoming_msg)

    assistant_msgs = [m for m in FakeMessage.instances if m is not incoming_msg]
    assert len(assistant_msgs) == 1
    reply = assistant_msgs[0]
    assert "مستندات بارگذاری‌شده موجود نیست" in reply.content
    assert reply.elements == []

    # Test invalid empty input handling
    FakeMessage.instances.clear()
    empty_msg = FakeMessage(content="   ")
    with patch("app.cl.Message", FakeMessage):
        await app.on_message(empty_msg)

    error_replies = [m for m in FakeMessage.instances if m is not empty_msg]
    assert len(error_replies) == 1
    assert error_replies[0].sent is True


@pytest.mark.asyncio
async def test_on_feedback_action_records_feedback() -> None:
    """Verify feedback action callback logs feedback and sends confirmation."""
    action = FakeAction(
        name="feedback_positive",
        payload={"trace_id": "trace-abc", "rating": "positive"},
    )
    action.remove = AsyncMock()  # type: ignore[attr-defined]

    with patch("app.cl.Message", FakeMessage):
        await app.on_feedback_action(action)

    assert len(FakeMessage.instances) == 1
    assert "ثبت شد" in FakeMessage.instances[0].content
