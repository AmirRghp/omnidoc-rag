"""LangGraph CRAG state definitions, typed contracts, and state validation."""

import uuid
from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast

from src.common.schemas import CitationItem, RetrievedChunk
from src.common.security import sanitize_query
from src.services.prompt_builder import detect_query_language

RelevanceStatus = Literal["correct", "ambiguous", "incorrect"]


class RAGState(TypedDict, total=False):
    """Complete state dictionary for LangGraph Corrective RAG (CRAG) workflow.

    Maintains query context, retrieved documents, relevance evaluation gates,
    iterative transformations, generated response, citations, and metadata.
    """

    # Query & Session Context
    query: str
    trace_id: str
    normalized_query: str
    language: Literal["fa", "en"]

    # Retrieval & Ranking Artifacts
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]

    # CRAG Evaluation & Grading
    relevance_status: RelevanceStatus
    crag_score: float

    # Iterative Transformation & Guardrails
    rewritten_query: str | None
    transformation_count: int

    # Generation & Delivery
    generation: str
    citations: list[CitationItem]
    refusal_reason: str | None
    cache_hit: bool
    error: str | None
    temperature: float
    enable_reflection: bool
    stream_tokens: bool


def create_initial_state(
    query: str,
    trace_id: str | None = None,
    language: Literal["fa", "en"] | None = None,
    temperature: float = 0.0,
    enable_reflection: bool = True,
    stream_tokens: bool = False,
) -> RAGState:
    """Create a fully-initialized RAGState dictionary with safe defaults.

    Args:
        query: Raw user query string.
        trace_id: Optional correlation/trace UUID string.
        language: Optional language override ("fa" or "en").
        temperature: LLM generation sampling temperature.
        enable_reflection: Whether CRAG query rewrite reflection is enabled.
        stream_tokens: Whether to stream tokens directly from vLLM during generation.

    Returns:
        Fully initialized RAGState dictionary.
    """
    try:
        clean_query = sanitize_query(query)
    except Exception:
        clean_query = query.strip()

    detected_lang = language or detect_query_language(clean_query)
    current_trace_id = trace_id or str(uuid.uuid4())

    return {
        "query": query,
        "trace_id": current_trace_id,
        "normalized_query": clean_query,
        "language": detected_lang,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "relevance_status": "ambiguous",
        "crag_score": 0.0,
        "rewritten_query": None,
        "transformation_count": 0,
        "generation": "",
        "citations": [],
        "refusal_reason": None,
        "cache_hit": False,
        "error": None,
        "temperature": temperature,
        "enable_reflection": enable_reflection,
        "stream_tokens": stream_tokens,
    }


def validate_rag_state(state: Mapping[str, Any]) -> RAGState:
    """Validate a dictionary against the RAGState contract.

    Args:
        state: State dictionary to validate.

    Returns:
        Validated RAGState dictionary.

    Raises:
        ValueError: If mandatory keys are missing or field values violate constraints.
    """
    if not isinstance(state, Mapping):
        raise ValueError(f"State must be a dictionary, got {type(state).__name__}")

    required_keys = {"query", "trace_id", "retrieved_chunks", "reranked_chunks"}
    missing = required_keys - state.keys()
    if missing:
        raise ValueError(f"Missing required keys in RAGState: {sorted(missing)}")

    if not isinstance(state.get("query"), str):
        raise ValueError("Field 'query' must be a string")

    if not isinstance(state.get("trace_id"), str):
        raise ValueError("Field 'trace_id' must be a string")

    if "crag_score" in state:
        score = state["crag_score"]
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            raise ValueError(f"Field 'crag_score' must be a float between 0.0 and 1.0, got {score}")

    if "relevance_status" in state and state["relevance_status"] is not None:
        status = state["relevance_status"]
        valid_statuses = {"correct", "ambiguous", "incorrect"}
        if status not in valid_statuses:
            raise ValueError(
                f"Field 'relevance_status' must be one of {valid_statuses}, got '{status}'"
            )

    if "transformation_count" in state:
        count = state["transformation_count"]
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                f"Field 'transformation_count' must be a non-negative int, got {count}"
            )

    return cast(RAGState, state)
