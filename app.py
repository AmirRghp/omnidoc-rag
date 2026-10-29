"""OmniDoc RAG Chainlit web application entrypoint, CRAG streaming, and citations drawer."""

import html
import os
import re
import uuid
from typing import Any

import chainlit as cl

from src.common.config import create_settings
from src.common.exceptions import InvalidInputSecurityError
from src.common.logger import configure_logger, get_logger, set_trace_id
from src.common.schemas import CitationItem, RetrievedChunk
from src.common.security import sanitize_query, tokenize_generation_stream
from src.graph.workflow import CRAGWorkflow
from src.services.cache import RedisQueryCache

logger = get_logger(__name__)

STEP_RETRIEVAL_NAME = "🔍 Retrieval (Dense + Sparse)"
STEP_RERANK_NAME = "⚖️ Fusion & Reranking"
STEP_CRAG_NAME = "🎯 Quality Gate (CRAG)"

REWRITE_ALERT_TEXT = "کوئری شما برای دقت بیشتر به عبارت تخصصی‌تر بازنویسی شد."

_workflow_instance: CRAGWorkflow | None = None

_DANGEROUS_HTML_RE = re.compile(
    r"<\s*/?\s*(?:script|iframe|object|embed|svg|style|link|meta|form|input|button|base)\b[^>]*>"
    r"|on\w+\s*=\s*[\"'][^\"']*[\"']"
    r"|javascript\s*:",
    re.IGNORECASE,
)


def sanitize_markdown_html(text: str) -> str:
    """Escape dangerous HTML/XSS constructs in LLM output while preserving Markdown and code."""
    if not text:
        return ""
    return _DANGEROUS_HTML_RE.sub(lambda m: html.escape(m.group(0)), text)


def get_crag_workflow() -> CRAGWorkflow:
    """Return or lazily initialize the singleton CRAGWorkflow instance."""
    global _workflow_instance
    if _workflow_instance is None:
        settings = create_settings()
        cache = RedisQueryCache(settings=settings) if settings.semantic_cache_enabled else None
        _workflow_instance = CRAGWorkflow(settings=settings, cache=cache)
    return _workflow_instance


def format_retrieval_step_output(
    retrieved_chunks: list[RetrievedChunk],
    query: str,
) -> str:
    """Format the collapsible Retrieval (Dense + Sparse) step summary.

    Args:
        retrieved_chunks: Hybrid candidates retrieved via Dense + BM25 search.
        query: Active search query executed against the indexes.

    Returns:
        Markdown summary string for the retrieval step body.
    """
    unique_sources: list[str] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        if chunk.source_file not in seen:
            seen.add(chunk.source_file)
            unique_sources.append(chunk.source_file)

    sources_line = ", ".join(f"`{src}`" for src in unique_sources) if unique_sources else "None"
    return (
        f"- **Active Query:** `{query}`\n"
        f"- **Candidates Retrieved:** `{len(retrieved_chunks)}` chunks\n"
        f"- **Matched Sources:** {sources_line}"
    )


def format_rerank_step_output(reranked_chunks: list[RetrievedChunk]) -> str:
    """Format the collapsible Fusion & Reranking step summary.

    Args:
        reranked_chunks: Top-N candidates rescored by Cross-Encoder.

    Returns:
        Markdown summary string listing ranked chunks, chunk IDs, and scores.
    """
    if not reranked_chunks:
        return "No candidates passed fusion and reranking (هیچ قطعه‌ای یافت نشد)."

    lines: list[str] = []
    for idx, chunk in enumerate(reranked_chunks, start=1):
        heading = ""
        if isinstance(chunk.metadata, dict):
            raw_heading = chunk.metadata.get("section_heading", "")
            if raw_heading:
                heading = f" — *{raw_heading}*"
        lines.append(
            f"{idx}. `{chunk.source_file}#chunk-{chunk.chunk_id}` "
            f"(Score: `{chunk.score:.3f}`){heading}"
        )
    return "\n".join(lines)


def format_crag_step_output(
    status: str,
    score: float,
    rewritten_query: str | None = None,
) -> str:
    """Format the collapsible Quality Gate (CRAG) step output.

    Args:
        status: Relevance status ('correct', 'ambiguous', or 'incorrect').
        score: Maximum calibrated Cross-Encoder relevance score.
        rewritten_query: Optional refined query when status is ambiguous.

    Returns:
        Markdown summary string for the CRAG evaluation step.
    """
    normalized_status = status.strip().upper() or "INCORRECT"
    lines = [
        f"- **Decision:** `{normalized_status}`",
        f"- **Confidence Score:** `{score:.3f}`",
    ]
    if rewritten_query:
        lines.append(f"- **Notice:** {REWRITE_ALERT_TEXT}")
        lines.append(f"- **Refined Query:** `{rewritten_query}`")
    return "\n".join(lines)


def format_citation_label(
    index: int,
    source_file: str,
    chunk_id: str | int | None = None,
) -> str:
    """Build a deterministic LTR citation identifier matching side-drawer elements.

    Args:
        index: 1-based citation index.
        source_file: Source Markdown filename.
        chunk_id: Optional chunk identifier within the source document.

    Returns:
        Formatted citation label string (e.g., '[1] docker_cli_run.md#chunk-0').
    """
    suffix = f"#chunk-{chunk_id}" if chunk_id is not None else ""
    return f"[{index}] {source_file}{suffix}"


def _resolve_chunk_id(
    index: int,
    citation: CitationItem,
    chunks: list[RetrievedChunk] | None,
) -> str | int:
    """Resolve the chunk_id corresponding to a CitationItem."""
    if chunks:
        if 0 <= index < len(chunks) and chunks[index].source_file == citation.source_file:
            return chunks[index].chunk_id
        for chunk in chunks:
            if chunk.source_file == citation.source_file:
                return chunk.chunk_id
    return index


def format_citation_pills_markdown(
    citations: list[CitationItem],
    chunks: list[RetrievedChunk] | None = None,
) -> str:
    """Render HTML/Markdown citation pills referencing side-drawer element names.

    Args:
        citations: Grounded citations emitted by the generation node.
        chunks: Optional reranked chunks to resolve exact chunk_id values.

    Returns:
        Markdown/HTML footer string with `.citation-pill` badges, or empty string.
    """
    if not citations:
        return ""

    pills: list[str] = []
    for idx, cit in enumerate(citations):
        chunk_id = _resolve_chunk_id(idx, cit, chunks)
        safe_label = html.escape(format_citation_label(idx + 1, cit.source_file, chunk_id))
        pills.append(
            f'<span class="citation-pill" dir="ltr" role="button" tabindex="0" '
            f'aria-label="Citation {safe_label}">{safe_label} ({cit.relevance_score:.2f})</span>'
        )

    pills_row = " ".join(pills)
    return f"\n\n---\n**منابع (Citations):**\n{pills_row}"


def build_citation_elements(
    citations: list[CitationItem],
    chunks: list[RetrievedChunk] | None = None,
) -> list[cl.Text]:
    """Construct clickable side-drawer Chainlit Text elements for each citation.

    Args:
        citations: List of CitationItem models from RAGState.
        chunks: Optional list of RetrievedChunk models for chunk_id lookup.

    Returns:
        List of `cl.Text` elements configured with `display="side"`.
    """
    elements: list[cl.Text] = []
    for idx, cit in enumerate(citations):
        chunk_id = _resolve_chunk_id(idx, cit, chunks)
        label = format_citation_label(idx + 1, cit.source_file, chunk_id)
        drawer_body = (
            f"### `{cit.source_file}`\n"
            f"- **Chunk ID:** `chunk-{chunk_id}`\n"
            f"- **Section:** {cit.section_heading or 'N/A'}\n"
            f"- **Relevance Score:** `{cit.relevance_score:.3f}`\n\n"
            f"#### Excerpt\n"
            f"```markdown\n{cit.snippet}\n```"
        )
        elements.append(
            cl.Text(
                name=label,
                content=drawer_body,
                display="side",
            )
        )
    return elements


def build_feedback_actions(trace_id: str) -> list[cl.Action]:
    """Build thumbs-up and thumbs-down feedback action buttons for a response.

    Args:
        trace_id: Correlation trace ID for the query execution.

    Returns:
        List of two `cl.Action` buttons ('feedback_positive', 'feedback_negative').
    """
    return [
        cl.Action(
            name="feedback_positive",
            payload={"trace_id": trace_id, "rating": "positive"},
            label="👍 مفید بود",
            tooltip="پاسخ دقیق و مستند بود",
        ),
        cl.Action(
            name="feedback_negative",
            payload={"trace_id": trace_id, "rating": "negative"},
            label="👎 نیاز به بهبود",
            tooltip="پاسخ ناقص یا نامرتبط بود",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize user chat session and display welcome guidance."""
    configure_logger(log_level="INFO", json_format=True)
    session_id = str(uuid.uuid4())
    try:
        cl.user_session.set("session_id", session_id)  # type: ignore[no-untyped-call]
    except Exception:
        logger.debug("chainlit_user_session_unavailable_outside_context")

    logger.info("chainlit_chat_session_started", session_id=session_id)

    welcome_content = (
        "سلام! به دستیار فنی **OmniDoc RAG** خوش آمدید.\n\n"
        "می‌توانید سوالات تخصصی خود درباره **Docker**، **PostgreSQL** یا **Linux** را "
        "به زبان فارسی یا انگلیسی مطرح کنید.\n\n"
        "> برای مثال:\n"
        "> `چگونه پورت‌های داکر را به صورت داینامیک نگاشت کنیم؟`"
    )

    await cl.Message(
        content=welcome_content,
        author="OmniDoc Assistant",
    ).send()  # type: ignore[no-untyped-call]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Stream LangGraph CRAG reasoning steps, grounded tokens, and citations drawer."""
    raw_query = message.content if message.content else ""
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)

    try:
        clean_query = sanitize_query(raw_query)
    except InvalidInputSecurityError as exc:
        logger.warning("chainlit_invalid_query", trace_id=trace_id, error=str(exc))
        await cl.Message(
            content=(
                "پرسش واردشده معتبر نیست. لطفاً متنی بین ۱ تا ۱۰۰۰ کاراکتر وارد کنید.\n\n"
                f"`{exc.message}`"
            ),
            author="OmniDoc Assistant",
        ).send()  # type: ignore[no-untyped-call]
        return

    session_id = trace_id
    try:
        stored_session = cl.user_session.get("session_id")  # type: ignore[no-untyped-call]
        if isinstance(stored_session, str) and stored_session:
            session_id = stored_session
    except Exception:
        pass

    logger.info(
        "chainlit_message_received",
        trace_id=trace_id,
        session_id=session_id,
        query_len=len(clean_query),
    )

    workflow = get_crag_workflow()
    active_query = clean_query
    latest_reranked: list[RetrievedChunk] = []
    latest_crag_score = 0.0
    latest_status = "incorrect"
    rewritten_query: str | None = None
    final_citations: list[CitationItem] = []

    response_msg = cl.Message(content="", author="OmniDoc Assistant")
    await response_msg.send()  # type: ignore[no-untyped-call]

    try:
        async for event in workflow.astream(
            query=clean_query,
            trace_id=trace_id,
            thread_id=session_id,
        ):
            if "retrieve" in event:
                node_data: dict[str, Any] = event["retrieve"]
                retrieved_chunks: list[RetrievedChunk] = node_data.get("retrieved_chunks", [])
                latest_reranked = node_data.get("reranked_chunks", [])

                async with cl.Step(name=STEP_RETRIEVAL_NAME, type="retrieval") as step_ret:
                    step_ret.input = active_query
                    step_ret.output = format_retrieval_step_output(retrieved_chunks, active_query)

                async with cl.Step(name=STEP_RERANK_NAME, type="rerank") as step_rerank:
                    step_rerank.input = f"{len(retrieved_chunks)} candidates"
                    step_rerank.output = format_rerank_step_output(latest_reranked)

            elif "grade_documents" in event:
                node_data = event["grade_documents"]
                latest_crag_score = float(node_data.get("crag_score", 0.0))
                latest_status = str(node_data.get("relevance_status", "incorrect"))

                async with cl.Step(name=STEP_CRAG_NAME, type="tool") as step_crag:
                    step_crag.input = f"Top Score: {latest_crag_score:.3f}"
                    step_crag.output = format_crag_step_output(
                        status=latest_status,
                        score=latest_crag_score,
                        rewritten_query=rewritten_query,
                    )

            elif "rewrite_query" in event:
                node_data = event["rewrite_query"]
                rewritten_query = str(node_data.get("rewritten_query", "")).strip()
                if rewritten_query:
                    active_query = rewritten_query
                notice_block = (
                    f"> 💡 **{REWRITE_ALERT_TEXT}**\n> `Rewritten Query: {active_query}`\n\n"
                )
                await response_msg.stream_token(notice_block)

            elif "generate" in event:
                node_data = event["generate"]
                generation_text = sanitize_markdown_html(str(node_data.get("generation", "")))
                raw_citations = node_data.get("citations", [])
                final_citations = [
                    c if isinstance(c, CitationItem) else CitationItem.model_validate(c)
                    for c in raw_citations
                ]
                for token in tokenize_generation_stream(generation_text):
                    await response_msg.stream_token(token)

            elif "refuse" in event:
                node_data = event["refuse"]
                refusal_text = sanitize_markdown_html(str(node_data.get("generation", "")))
                final_citations = []
                for token in tokenize_generation_stream(refusal_text):
                    await response_msg.stream_token(token)

        if final_citations:
            citations_footer = format_citation_pills_markdown(
                final_citations, chunks=latest_reranked
            )
            await response_msg.stream_token(citations_footer)
            response_msg.elements = build_citation_elements(  # type: ignore[assignment]
                final_citations, chunks=latest_reranked
            )

        response_msg.actions = build_feedback_actions(trace_id)
        await response_msg.update()  # type: ignore[no-untyped-call]

    except Exception as exc:
        logger.error(
            "chainlit_stream_error",
            trace_id=trace_id,
            error=str(exc),
            exc_info=True,
        )
        await response_msg.stream_token(
            "\n\n⚠️ خطایی در پردازش درخواست رخ داد. لطفاً دوباره تلاش کنید."
        )
        await response_msg.update()  # type: ignore[no-untyped-call]


@cl.action_callback("feedback_positive")
@cl.action_callback("feedback_negative")
async def on_feedback_action(action: cl.Action) -> None:
    """Handle user thumbs-up / thumbs-down feedback button clicks."""
    payload = action.payload if isinstance(action.payload, dict) else {}
    trace_id = str(payload.get("trace_id", "unknown"))
    rating = str(payload.get("rating", action.name))

    logger.info(
        "chainlit_user_feedback_recorded",
        trace_id=trace_id,
        rating=rating,
    )

    await cl.Message(
        content="سپاس! بازخورد شما با موفقیت ثبت شد.",
        author="OmniDoc Assistant",
    ).send()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    import uvicorn
    from chainlit.server import app as chainlit_server_app

    port = int(os.environ.get("PORT", "3000"))
    uvicorn.run(chainlit_server_app, host="0.0.0.0", port=port)
