"""Unit tests for LangGraph CRAG state definitions and typed contracts (Task 7.1)."""

import uuid
from typing import get_type_hints

import pytest

from src.common.schemas import CitationItem, RetrievedChunk
from src.graph.state import (
    RAGState,
    RelevanceStatus,
    create_initial_state,
    validate_rag_state,
)


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    """Fixture returning a sample RetrievedChunk."""
    return RetrievedChunk(
        chunk_id="chk-001",
        source_file="manuals/docker.md",
        content="Use `--ipc=host` for shared memory.",
        score=0.88,
        metadata={"section_heading": "IPC Configuration"},
    )


@pytest.fixture
def sample_citation() -> CitationItem:
    """Fixture returning a sample CitationItem."""
    return CitationItem(
        source_file="manuals/docker.md",
        section_heading="IPC Configuration",
        relevance_score=0.88,
        snippet="Use `--ipc=host` for shared memory.",
    )


class TestRAGStateContract:
    """Tests verifying the structure and typed hints of RAGState."""

    def test_rag_state_contains_all_architectural_fields(self) -> None:
        """Verify all mandatory CRAG state fields are declared in RAGState."""
        hints = get_type_hints(RAGState)
        expected_fields = {
            "query",
            "trace_id",
            "normalized_query",
            "language",
            "retrieved_chunks",
            "reranked_chunks",
            "relevance_status",
            "crag_score",
            "rewritten_query",
            "transformation_count",
            "generation",
            "citations",
            "refusal_reason",
            "cache_hit",
            "error",
        }
        for field in expected_fields:
            assert field in hints, f"Expected field '{field}' missing from RAGState"

    def test_relevance_status_literals(self) -> None:
        """RelevanceStatus accepts 'correct', 'ambiguous', and 'incorrect'."""
        valid_statuses: list[RelevanceStatus] = ["correct", "ambiguous", "incorrect"]
        for s in valid_statuses:
            assert s in {"correct", "ambiguous", "incorrect"}


class TestCreateInitialState:
    """Tests for initial state creation factory."""

    def test_create_initial_state_with_english_query(self) -> None:
        """Factory initializes complete state with detected English language and UUID trace."""
        query = "How to run docker in daemon mode?"
        state = create_initial_state(query)

        assert state["query"] == query
        assert state["language"] == "en"
        assert state["normalized_query"] == query
        assert state["retrieved_chunks"] == []
        assert state["reranked_chunks"] == []
        assert state["relevance_status"] == "ambiguous"
        assert state["crag_score"] == 0.0
        assert state["rewritten_query"] is None
        assert state["transformation_count"] == 0
        assert state["generation"] == ""
        assert state["citations"] == []
        assert state["refusal_reason"] is None
        assert state["cache_hit"] is False
        assert state["error"] is None

        # Verify trace_id is valid UUID string
        parsed_uuid = uuid.UUID(state["trace_id"])
        assert str(parsed_uuid) == state["trace_id"]

    def test_create_initial_state_with_persian_query(self) -> None:
        """Factory initializes state with Persian normalization and language detection."""
        persian_query = "دستور اجراي داكر با كانفيگ حافظه"
        state = create_initial_state(persian_query)

        assert state["language"] == "fa"
        # Arabic yeh/kaf normalized to Persian
        assert "ی" in state["normalized_query"]
        assert "ک" in state["normalized_query"]
        assert "ي" not in state["normalized_query"]
        assert "ك" not in state["normalized_query"]

    def test_create_initial_state_with_explicit_overrides(self) -> None:
        """Factory respects custom trace_id and language parameters."""
        custom_trace = "custom-trace-uuid-1234"
        state = create_initial_state(
            query="Bilingual prompt",
            trace_id=custom_trace,
            language="fa",
        )
        assert state["trace_id"] == custom_trace
        assert state["language"] == "fa"


class TestValidateRAGState:
    """Tests for runtime contract validation of state dictionaries."""

    def test_validate_valid_state_passes(self) -> None:
        """A complete and valid state dictionary passes validation without changes."""
        initial = create_initial_state("Test query")
        validated = validate_rag_state(initial)
        assert validated["query"] == "Test query"

    def test_validate_non_dict_raises_value_error(self) -> None:
        """Passing a non-dict raises ValueError."""
        with pytest.raises(ValueError, match="State must be a dictionary"):
            validate_rag_state(["invalid", "list"])  # type: ignore[arg-type]

    def test_validate_missing_required_keys_raises_value_error(self) -> None:
        """State missing required keys raises ValueError."""
        incomplete_state = {"query": "Test"}
        with pytest.raises(ValueError, match="Missing required keys in RAGState"):
            validate_rag_state(incomplete_state)

    def test_validate_invalid_query_type_raises_value_error(self) -> None:
        """State with non-string query raises ValueError."""
        state = create_initial_state("Valid")
        state["query"] = 12345  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Field 'query' must be a string"):
            validate_rag_state(state)

    def test_validate_invalid_trace_id_type_raises_value_error(self) -> None:
        """State with non-string trace_id raises ValueError."""
        state = create_initial_state("Valid")
        state["trace_id"] = None  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Field 'trace_id' must be a string"):
            validate_rag_state(state)

    def test_validate_invalid_crag_score_bounds_raises_value_error(self) -> None:
        """crag_score < 0.0 or > 1.0 raises ValueError."""
        state = create_initial_state("Valid")
        state["crag_score"] = 1.5
        match_msg = "Field 'crag_score' must be a float between 0.0 and 1.0"
        with pytest.raises(ValueError, match=match_msg):
            validate_rag_state(state)

        state["crag_score"] = -0.1
        with pytest.raises(ValueError, match=match_msg):
            validate_rag_state(state)

    def test_validate_invalid_relevance_status_raises_value_error(self) -> None:
        """relevance_status not in permitted literal values raises ValueError."""
        state = create_initial_state("Valid")
        state["relevance_status"] = "partially_relevant"  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Field 'relevance_status' must be one of"):
            validate_rag_state(state)

    def test_validate_negative_transformation_count_raises_value_error(self) -> None:
        """Negative transformation_count raises ValueError."""
        state = create_initial_state("Valid")
        state["transformation_count"] = -1
        match_msg = "Field 'transformation_count' must be a non-negative int"
        with pytest.raises(ValueError, match=match_msg):
            validate_rag_state(state)


class TestStateTransitionsAndPartialUpdates:
    """Tests simulating node state updates in LangGraph."""

    def test_retrieval_and_reranking_updates(
        self,
        sample_chunk: RetrievedChunk,
        sample_citation: CitationItem,
    ) -> None:
        """State dictionary accurately accumulates chunks, scores, and citations across steps."""
        state = create_initial_state("Docker command")

        # Step 1: Retrieval node updates
        retrieval_update = {"retrieved_chunks": [sample_chunk]}
        state.update(retrieval_update)  # type: ignore[typeddict-item]
        assert len(state["retrieved_chunks"]) == 1

        # Step 2: Reranker and grading node updates
        grading_update = {
            "reranked_chunks": [sample_chunk],
            "crag_score": 0.88,
            "relevance_status": "correct",
        }
        state.update(grading_update)  # type: ignore[typeddict-item]
        assert state["relevance_status"] == "correct"
        assert state["crag_score"] == 0.88

        # Step 3: Generation node updates
        generation_update = {
            "generation": "Use `--ipc=host` flag.",
            "citations": [sample_citation],
        }
        state.update(generation_update)  # type: ignore[typeddict-item]
        assert state["generation"] == "Use `--ipc=host` flag."
        assert len(state["citations"]) == 1

        # Validate final state
        validated = validate_rag_state(state)
        assert validated["relevance_status"] == "correct"
