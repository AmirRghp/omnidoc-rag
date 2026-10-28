"""Atomic execution nodes for the LangGraph Corrective RAG (CRAG) control plane."""

import asyncio
from typing import Any

from src.common.config import Settings, create_settings
from src.common.logger import get_logger
from src.common.schemas import CitationItem, RetrievedChunk
from src.graph.state import RAGState, RelevanceStatus
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import afuse_dense_and_sparse
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.sparse import SparseRetriever
from src.services.llm_client import VLLMClient
from src.services.prompt_builder import PromptBuilder, detect_query_language

logger = get_logger(__name__)


class CRAGNodes:
    """Encapsulates atomic CRAG workflow nodes with dependency injection."""

    def __init__(
        self,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        llm_client: VLLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize CRAG nodes with dependencies.

        Args:
            dense_retriever: Optional DenseRetriever instance.
            sparse_retriever: Optional SparseRetriever instance.
            reranker: Optional CrossEncoderReranker instance.
            llm_client: Optional VLLMClient instance.
            prompt_builder: Optional PromptBuilder instance.
            settings: Optional application settings.
        """
        self.settings = settings or create_settings()
        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()

    @property
    def dense(self) -> DenseRetriever:
        """Retrieve or lazily initialize DenseRetriever."""
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(settings=self.settings)
        return self._dense_retriever

    @property
    def sparse(self) -> SparseRetriever:
        """Retrieve or lazily initialize SparseRetriever."""
        if self._sparse_retriever is None:
            self._sparse_retriever = SparseRetriever()
        return self._sparse_retriever

    @property
    def ranker(self) -> CrossEncoderReranker:
        """Retrieve or lazily initialize CrossEncoderReranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(settings=self.settings)
        return self._reranker

    @property
    def llm(self) -> VLLMClient:
        """Retrieve or lazily initialize VLLMClient."""
        if self._llm_client is None:
            self._llm_client = VLLMClient(settings=self.settings)
        return self._llm_client

    async def retrieve_node(self, state: RAGState) -> dict[str, Any]:
        """Execute hybrid retrieval (Dense + Sparse with RRF) and Cross-Encoder reranking.

        Args:
            state: Current RAGState.

        Returns:
            State update dictionary with retrieved_chunks and reranked_chunks.
        """
        query = state.get("rewritten_query") or state.get("query", "")
        if not query or not query.strip():
            logger.warning("Empty query in retrieve_node; returning empty chunks")
            return {
                "retrieved_chunks": [],
                "reranked_chunks": [],
            }

        try:
            # 1. Dual parallel retrieval (Dense + Sparse)
            dense_results, sparse_results = await asyncio.gather(
                self.dense.asearch(query, top_k=self.settings.top_k_dense),
                self.sparse.asearch(query, top_k=self.settings.top_k_sparse),
            )

            # 2. Reciprocal Rank Fusion
            candidates: list[RetrievedChunk] = await afuse_dense_and_sparse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                k=self.settings.rrf_k,
                top_n=max(self.settings.rerank_top_n * 3, 15),
            )

            # 3. Cross-Encoder reranking to Top-N
            reranked: list[RetrievedChunk] = []
            if candidates:
                reranked = await self.ranker.arerank(
                    query=query,
                    chunks=candidates,
                    top_n=self.settings.rerank_top_n,
                )

            return {
                "retrieved_chunks": candidates,
                "reranked_chunks": reranked,
            }
        except Exception as exc:
            logger.error("Error during retrieval in retrieve_node: %s", exc, exc_info=True)
            return {
                "retrieved_chunks": [],
                "reranked_chunks": [],
                "error": str(exc),
            }

    async def grade_documents_node(self, state: RAGState) -> dict[str, Any]:
        """Evaluate top candidate relevance score against CRAG gating thresholds.

        Thresholds:
            score >= crag_correct_threshold (0.70): 'correct' -> generate
            crag_ambiguous_threshold (0.35) <= score < 0.70:
                if transformation_count < max_query_transformations: 'ambiguous' -> rewrite
                else: 'incorrect' -> refuse
            score < crag_ambiguous_threshold (0.35): 'incorrect' -> refuse

        Args:
            state: Current RAGState.

        Returns:
            State update dictionary with crag_score and relevance_status.
        """
        reranked_chunks = state.get("reranked_chunks", [])
        transformation_count = state.get("transformation_count", 0)

        if not reranked_chunks:
            logger.info("No documents to grade; marking relevance_status as incorrect")
            return {
                "crag_score": 0.0,
                "relevance_status": "incorrect",
            }

        max_score = max(chunk.score for chunk in reranked_chunks)

        if max_score >= self.settings.crag_correct_threshold:
            status: RelevanceStatus = "correct"
        elif max_score >= self.settings.crag_ambiguous_threshold:
            if transformation_count < self.settings.max_query_transformations:
                status = "ambiguous"
            else:
                logger.info(
                    "Query ambiguous (score=%.2f) but reached max transformations (%d); "
                    "routing to refuse",
                    max_score,
                    self.settings.max_query_transformations,
                )
                status = "incorrect"
        else:
            status = "incorrect"

        logger.info(
            "Graded documents: max_score=%.3f, transformation_count=%d -> status=%s",
            max_score,
            transformation_count,
            status,
        )
        return {
            "crag_score": float(max_score),
            "relevance_status": status,
        }

    async def rewrite_query_node(self, state: RAGState) -> dict[str, Any]:
        """Refine ambiguous query using LLM and increment transformation counter.

        Args:
            state: Current RAGState.

        Returns:
            State update dictionary with rewritten_query and incremented transformation_count.
        """
        original_query = state.get("query", "")
        current_count = state.get("transformation_count", 0)
        new_count = current_count + 1

        try:
            messages = self.prompt_builder.build_rewrite_messages(original_query)
            raw_rewritten = await self.llm.generate(messages, temperature=0.0)
            rewritten_clean = raw_rewritten.strip().strip('"').strip("'").strip()
            if not rewritten_clean:
                rewritten_clean = original_query

            logger.info(
                "Rewrote query (attempt %d): '%s' -> '%s'",
                new_count,
                original_query,
                rewritten_clean,
            )
            return {
                "rewritten_query": rewritten_clean,
                "transformation_count": new_count,
            }
        except Exception as exc:
            logger.warning("Failed to rewrite query: %s. Reusing original query.", exc)
            return {
                "rewritten_query": original_query,
                "transformation_count": new_count,
                "error": str(exc),
            }

    async def generate_node(self, state: RAGState) -> dict[str, Any]:
        """Generate grounded technical answer using strict XML context enclosure.

        Args:
            state: Current RAGState.

        Returns:
            State update dictionary with generation, citations, and relevance_status.
        """
        query = state.get("query", "")
        chunks = state.get("reranked_chunks", [])

        citations: list[CitationItem] = []
        for chunk in chunks:
            heading = ""
            if isinstance(chunk.metadata, dict):
                heading = str(chunk.metadata.get("section_heading", ""))
            citations.append(
                CitationItem(
                    source_file=chunk.source_file,
                    section_heading=heading,
                    relevance_score=float(chunk.score),
                    snippet=chunk.content[:200],
                )
            )

        try:
            messages = self.prompt_builder.build_generation_messages(query=query, chunks=chunks)
            raw_answer = await self.llm.generate(messages, temperature=0.0)
            validated_answer = self.prompt_builder.validate_output(raw_answer)

            return {
                "generation": validated_answer,
                "citations": citations,
                "relevance_status": "correct",
            }
        except Exception as exc:
            logger.error("Generation failed or blocked by output guardrails: %s", exc)
            language = state.get("language") or detect_query_language(query)
            refusal = self.prompt_builder.get_refusal_message(language)
            return {
                "generation": refusal,
                "citations": [],
                "relevance_status": "incorrect",
                "refusal_reason": f"generation_error: {exc}",
                "error": str(exc),
            }

    async def refuse_node(self, state: RAGState) -> dict[str, Any]:
        """Emit standardized polite bilingual refusal message without calling LLM generator.

        Args:
            state: Current RAGState.

        Returns:
            State update dictionary with polite refusal generation and empty citations.
        """
        query = state.get("query", "")
        language = state.get("language")
        if not language and query:
            language = detect_query_language(query)
        elif not language:
            language = "fa"

        refusal_message = self.prompt_builder.get_refusal_message(language)
        reason = state.get("refusal_reason") or "insufficient_context"

        logger.info(
            "Emitted refusal message (language=%s, reason=%s)",
            language,
            reason,
        )
        return {
            "generation": refusal_message,
            "citations": [],
            "relevance_status": "incorrect",
            "refusal_reason": reason,
        }
