"""LangGraph Corrective RAG (CRAG) workflow assembly, routing logic, and compilation."""

from collections.abc import AsyncIterator
from functools import partial
from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.common.config import Settings, create_settings
from src.common.logger import get_logger
from src.graph.nodes import CRAGNodes
from src.graph.state import RAGState, create_initial_state
from src.services.cache import RedisQueryCache

logger = get_logger(__name__)

NodeName = Literal[
    "retrieve",
    "grade_documents",
    "rewrite_query",
    "generate",
    "refuse",
]
RouteDestination = Literal["generate", "rewrite_query", "refuse"]


def route_after_grading(
    state: RAGState,
    max_transformations: int = 1,
) -> RouteDestination:
    """Determine next workflow node after document relevance grading.

    Routing rules:
        - 'correct' -> 'generate'
        - 'ambiguous' -> 'rewrite_query' if transformation_count < max_transformations else 'refuse'
        - 'incorrect' -> 'refuse'

    Args:
        state: Current RAGState dictionary.
        max_transformations: Maximum allowed query rewrite cycles (default: 1).

    Returns:
        Target node name ('generate', 'rewrite_query', or 'refuse').
    """
    raw_status = str(state.get("relevance_status", "incorrect")).strip().lower()
    transformation_count = int(state.get("transformation_count", 0))
    enable_reflection = bool(state.get("enable_reflection", True))

    if raw_status == "correct":
        return "generate"

    if raw_status == "ambiguous":
        if enable_reflection and transformation_count < max_transformations:
            return "rewrite_query"
        logger.info(
            "Routing ambiguous query to refuse: transformation_count=%d, max=%d, reflection=%s",
            transformation_count,
            max_transformations,
            enable_reflection,
        )
        return "refuse"

    return "refuse"


def create_crag_state_graph(
    nodes: CRAGNodes | None = None,
    settings: Settings | None = None,
) -> StateGraph[Any]:
    """Assemble the uncompiled LangGraph StateGraph for Corrective RAG.

    Args:
        nodes: Optional CRAGNodes instance (created with settings if omitted).
        settings: Optional Settings instance.

    Returns:
        Configured StateGraph(RAGState) with all nodes and edges registered.
    """
    active_settings = settings or (nodes.settings if nodes is not None else create_settings())
    active_nodes = nodes or CRAGNodes(settings=active_settings)

    graph: StateGraph[Any] = StateGraph(RAGState)

    # Register atomic execution nodes
    graph.add_node("retrieve", active_nodes.retrieve_node)
    graph.add_node("grade_documents", active_nodes.grade_documents_node)
    graph.add_node("rewrite_query", active_nodes.rewrite_query_node)
    graph.add_node("generate", active_nodes.generate_node)
    graph.add_node("refuse", active_nodes.refuse_node)

    # Linear entry and retrieval -> grading flow
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # Conditional routing out of grade_documents
    router_fn = partial(
        route_after_grading,
        max_transformations=active_settings.max_query_transformations,
    )
    graph.add_conditional_edges(
        "grade_documents",
        router_fn,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "refuse": "refuse",
        },
    )

    # Corrective loop back to retrieve after query rewrite, and terminal edges
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph


def build_crag_workflow(
    nodes: CRAGNodes | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver[Any] | bool | None = None,
) -> CompiledStateGraph[Any]:
    """Build and compile the Corrective RAG LangGraph workflow.

    Args:
        nodes: Optional CRAGNodes instance with injected dependencies.
        settings: Optional application settings.
        checkpointer: Optional checkpointer instance, True to use an in-memory
            MemorySaver, or None/False to compile without state checkpointing.

    Returns:
        CompiledStateGraph ready for `.ainvoke()` or `.astream()`.
    """
    graph = create_crag_state_graph(nodes=nodes, settings=settings)

    resolved_checkpointer: BaseCheckpointSaver[Any] | None = None
    if isinstance(checkpointer, BaseCheckpointSaver):
        resolved_checkpointer = checkpointer
    elif checkpointer is True:
        resolved_checkpointer = MemorySaver()

    if resolved_checkpointer is not None:
        return graph.compile(checkpointer=resolved_checkpointer)
    return graph.compile()


class CRAGWorkflow:
    """High-level asynchronous runner for the compiled CRAG workflow."""

    def __init__(
        self,
        nodes: CRAGNodes | None = None,
        settings: Settings | None = None,
        checkpointer: BaseCheckpointSaver[Any] | bool | None = None,
        cache: RedisQueryCache | None = None,
    ) -> None:
        """Initialize the CRAGWorkflow facade.

        Args:
            nodes: Optional CRAGNodes instance.
            settings: Optional Settings instance.
            checkpointer: Optional checkpointer or boolean flag for MemorySaver.
            cache: Optional two-tier RedisQueryCache for exact/semantic caching.
        """
        self.settings = settings or (nodes.settings if nodes is not None else create_settings())
        self.nodes = nodes or CRAGNodes(settings=self.settings)
        self.cache = cache
        self.app: CompiledStateGraph[Any] = build_crag_workflow(
            nodes=self.nodes,
            settings=self.settings,
            checkpointer=checkpointer,
        )

    @staticmethod
    def _build_config(thread_id: str | None) -> RunnableConfig | None:
        """Construct RunnableConfig when a thread_id is provided."""
        if not thread_id:
            return None
        return cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})

    async def ainvoke(
        self,
        query: str,
        trace_id: str | None = None,
        language: Literal["fa", "en"] | None = None,
        thread_id: str | None = None,
        temperature: float = 0.0,
        enable_reflection: bool = True,
    ) -> RAGState:
        """Execute the full CRAG workflow asynchronously and return the final state.

        Args:
            query: User technical query.
            trace_id: Optional correlation trace ID.
            language: Optional language override ('fa' or 'en').
            thread_id: Optional checkpoint thread identifier.
            temperature: LLM sampling temperature.
            enable_reflection: Whether CRAG query rewrite reflection is enabled.

        Returns:
            Final RAGState after graph completion.
        """
        initial_state = create_initial_state(
            query=query,
            trace_id=trace_id,
            language=language,
            temperature=temperature,
            enable_reflection=enable_reflection,
        )

        if self.cache is not None and self.settings.semantic_cache_enabled:
            cached = await self.cache.get(query)
            if cached is not None:
                initial_state["generation"] = cached.response
                initial_state["citations"] = list(cached.citations)
                initial_state["relevance_status"] = "correct"
                initial_state["crag_score"] = 1.0
                initial_state["cache_hit"] = True
                return initial_state

        config = self._build_config(thread_id or initial_state.get("trace_id"))
        result = cast(RAGState, await self.app.ainvoke(initial_state, config=config))

        if (
            self.cache is not None
            and self.settings.semantic_cache_enabled
            and result.get("relevance_status") == "correct"
            and result.get("generation")
        ):
            await self.cache.set(
                query=query,
                response=str(result["generation"]),
                citations=result.get("citations", []),
            )

        return result

    async def astream(
        self,
        query: str,
        trace_id: str | None = None,
        language: Literal["fa", "en"] | None = None,
        thread_id: str | None = None,
        temperature: float = 0.0,
        enable_reflection: bool = True,
        stream_tokens: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream node execution updates from the CRAG workflow.

        Args:
            query: User technical query.
            trace_id: Optional correlation trace ID.
            language: Optional language override ('fa' or 'en').
            thread_id: Optional checkpoint thread identifier.
            temperature: LLM sampling temperature.
            enable_reflection: Whether CRAG query rewrite reflection is enabled.
            stream_tokens: Whether to stream tokens directly from vLLM during generation.

        Yields:
            Dictionaries mapping executed node names to their state updates.
        """
        initial_state = create_initial_state(
            query=query,
            trace_id=trace_id,
            language=language,
            temperature=temperature,
            enable_reflection=enable_reflection,
            stream_tokens=stream_tokens,
        )

        if self.cache is not None and self.settings.semantic_cache_enabled:
            cached = await self.cache.get(query)
            if cached is not None:
                yield {"retrieve": {"retrieved_chunks": [], "reranked_chunks": []}}
                yield {"grade_documents": {"crag_score": 1.0, "relevance_status": "correct"}}
                yield {
                    "generate": {
                        "generation": cached.response,
                        "citations": list(cached.citations),
                        "relevance_status": "correct",
                        "cache_hit": True,
                    }
                }
                return

        config = self._build_config(thread_id or initial_state.get("trace_id"))
        async for event in self.app.astream(initial_state, config=config):
            if isinstance(event, dict):
                if (
                    "generate" in event
                    and self.cache is not None
                    and self.settings.semantic_cache_enabled
                ):
                    gen_data = event["generate"]
                    if (
                        isinstance(gen_data, dict)
                        and gen_data.get("relevance_status") == "correct"
                        and gen_data.get("generation")
                    ):
                        await self.cache.set(
                            query=query,
                            response=str(gen_data["generation"]),
                            citations=gen_data.get("citations", []),
                        )
                yield event
