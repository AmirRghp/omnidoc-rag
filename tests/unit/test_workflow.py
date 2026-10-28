"""Unit tests for LangGraph CRAG workflow assembly and conditional routing (Task 7.3)."""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from src.common.config import Settings
from src.common.schemas import RetrievedChunk
from src.graph.nodes import CRAGNodes
from src.graph.state import create_initial_state
from src.graph.workflow import (
    CRAGWorkflow,
    build_crag_workflow,
    create_crag_state_graph,
    route_after_grading,
)
from src.services.prompt_builder import PromptBuilder


@pytest.fixture
def mock_settings() -> Settings:
    """Provide isolated test settings for CRAG workflow."""
    return Settings(
        CRAG_CORRECT_THRESHOLD=0.70,
        CRAG_AMBIGUOUS_THRESHOLD=0.35,
        MAX_QUERY_TRANSFORMATIONS=1,
        TOP_K_DENSE=5,
        TOP_K_SPARSE=5,
        RRF_K=60,
        RERANK_TOP_N=3,
    )


@pytest.fixture
def high_score_chunk() -> RetrievedChunk:
    """Provide a chunk whose score exceeds crag_correct_threshold (0.70)."""
    return RetrievedChunk(
        chunk_id="chk-high",
        source_file="manuals/docker_ipc.md",
        content="Use `--ipc=host` to share the host IPC namespace across containers.",
        score=0.91,
        metadata={"section_heading": "IPC Namespace Configuration"},
    )


@pytest.fixture
def ambiguous_score_chunk() -> RetrievedChunk:
    """Provide a chunk with score in the ambiguous range [0.35, 0.70)."""
    return RetrievedChunk(
        chunk_id="chk-ambiguous",
        source_file="manuals/docker_overview.md",
        content="Docker containers isolate processes by default.",
        score=0.52,
        metadata={"section_heading": "Container Isolation"},
    )


@pytest.fixture
def low_score_chunk() -> RetrievedChunk:
    """Provide a chunk whose score falls below crag_ambiguous_threshold (0.35)."""
    return RetrievedChunk(
        chunk_id="chk-low",
        source_file="manuals/linux_misc.md",
        content="General Linux file permissions overview.",
        score=0.18,
        metadata={"section_heading": "File Permissions"},
    )


@pytest.fixture
def crag_nodes(mock_settings: Settings) -> CRAGNodes:
    """Provide CRAGNodes instance with mocked external dependencies."""
    return CRAGNodes(
        dense_retriever=AsyncMock(),
        sparse_retriever=AsyncMock(),
        reranker=AsyncMock(),
        llm_client=AsyncMock(),
        prompt_builder=PromptBuilder(),
        settings=mock_settings,
    )


class TestRouteAfterGrading:
    """Tests for conditional routing logic after grade_documents."""

    def test_routes_correct_to_generate(self) -> None:
        """Status 'correct' routes directly to 'generate'."""
        state = create_initial_state("How to configure Docker IPC?")
        state["relevance_status"] = "correct"
        state["transformation_count"] = 0

        assert route_after_grading(state) == "generate"

    def test_routes_ambiguous_to_rewrite_when_below_limit(self) -> None:
        """Status 'ambiguous' with transformation_count < 1 routes to 'rewrite_query'."""
        state = create_initial_state("Docker IPC issue")
        state["relevance_status"] = "ambiguous"
        state["transformation_count"] = 0

        assert route_after_grading(state, max_transformations=1) == "rewrite_query"

    def test_routes_ambiguous_to_refuse_when_limit_reached(self) -> None:
        """Status 'ambiguous' with transformation_count >= 1 routes to 'refuse'."""
        state = create_initial_state("Docker IPC issue")
        state["relevance_status"] = "ambiguous"
        state["transformation_count"] = 1

        assert route_after_grading(state, max_transformations=1) == "refuse"

    def test_routes_incorrect_to_refuse(self) -> None:
        """Status 'incorrect' routes directly to 'refuse'."""
        state = create_initial_state("Unrelated question")
        state["relevance_status"] = "incorrect"
        state["transformation_count"] = 0

        assert route_after_grading(state) == "refuse"

    def test_routes_uppercase_status_values_gracefully(self) -> None:
        """Handles uppercase status labels ('CORRECT', 'AMBIGUOUS', 'INCORRECT') safely."""
        state = create_initial_state("Test query")
        state["relevance_status"] = "CORRECT"  # type: ignore[typeddict-item]
        assert route_after_grading(state) == "generate"

        state["relevance_status"] = "AMBIGUOUS"  # type: ignore[typeddict-item]
        state["transformation_count"] = 0
        assert route_after_grading(state) == "rewrite_query"

        state["relevance_status"] = "INCORRECT"  # type: ignore[typeddict-item]
        assert route_after_grading(state) == "refuse"


class TestCRAGWorkflowAssembly:
    """Tests for compiled LangGraph StateGraph execution across all CRAG paths."""

    def test_create_crag_state_graph_registers_all_nodes(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
    ) -> None:
        """StateGraph contains retrieve, grade_documents, rewrite_query, generate, and refuse."""
        graph = create_crag_state_graph(nodes=crag_nodes, settings=mock_settings)
        expected_nodes = {
            "retrieve",
            "grade_documents",
            "rewrite_query",
            "generate",
            "refuse",
        }
        assert expected_nodes.issubset(set(graph.nodes.keys()))

    @pytest.mark.asyncio
    async def test_workflow_correct_path_generates_answer(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        high_score_chunk: RetrievedChunk,
    ) -> None:
        """High-relevance chunks route retrieve -> grade_documents -> generate."""
        crag_nodes.dense.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.sparse.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.ranker.arerank = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.llm.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="Pass `--ipc=host` to share the host IPC namespace."
        )

        app = build_crag_workflow(nodes=crag_nodes, settings=mock_settings)
        initial_state = create_initial_state("How to share host IPC in Docker?")
        final_state = await app.ainvoke(initial_state)

        assert final_state["relevance_status"] == "correct"
        assert final_state["crag_score"] == 0.91
        assert final_state["transformation_count"] == 0
        assert "--ipc=host" in final_state["generation"]
        assert len(final_state["citations"]) == 1
        assert final_state["citations"][0].source_file == "manuals/docker_ipc.md"

    @pytest.mark.asyncio
    async def test_workflow_ambiguous_path_rewrites_and_succeeds(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        ambiguous_score_chunk: RetrievedChunk,
        high_score_chunk: RetrievedChunk,
    ) -> None:
        """Ambiguous initial score triggers rewrite_query -> retrieve -> generate."""
        rerank_calls = 0

        async def mock_rerank(
            query: str,
            chunks: list[RetrievedChunk],
            top_n: int = 5,
        ) -> list[RetrievedChunk]:
            nonlocal rerank_calls
            rerank_calls += 1
            if rerank_calls == 1:
                return [ambiguous_score_chunk]
            return [high_score_chunk]

        llm_calls: list[str] = []

        async def mock_llm_generate(
            messages: list[dict[str, str]],
            temperature: float = 0.0,
            **kwargs: Any,
        ) -> str:
            llm_calls.append(messages[-1]["content"])
            if len(llm_calls) == 1:
                return "docker run --ipc=host shared memory configuration"
            return "Configure shared IPC using `docker run --ipc=host`."

        crag_nodes.dense.asearch = AsyncMock(  # type: ignore[method-assign]
            return_value=[ambiguous_score_chunk, high_score_chunk]
        )
        crag_nodes.sparse.asearch = AsyncMock(  # type: ignore[method-assign]
            return_value=[ambiguous_score_chunk, high_score_chunk]
        )
        crag_nodes.ranker.arerank = mock_rerank  # type: ignore[assignment,method-assign]
        crag_nodes.llm.generate = mock_llm_generate  # type: ignore[assignment,method-assign]

        app = build_crag_workflow(nodes=crag_nodes, settings=mock_settings)
        initial_state = create_initial_state("docker ipc")
        final_state = await app.ainvoke(initial_state)

        assert rerank_calls == 2
        assert final_state["transformation_count"] == 1
        assert final_state["rewritten_query"] == "docker run --ipc=host shared memory configuration"
        assert final_state["relevance_status"] == "correct"
        assert "docker run --ipc=host" in final_state["generation"]

    @pytest.mark.asyncio
    async def test_workflow_ambiguous_twice_refuses_after_max_rewrites(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        ambiguous_score_chunk: RetrievedChunk,
    ) -> None:
        """If query remains ambiguous after 1 rewrite, workflow routes to refuse."""
        crag_nodes.dense.asearch = AsyncMock(return_value=[ambiguous_score_chunk])  # type: ignore[method-assign]
        crag_nodes.sparse.asearch = AsyncMock(return_value=[ambiguous_score_chunk])  # type: ignore[method-assign]
        crag_nodes.ranker.arerank = AsyncMock(return_value=[ambiguous_score_chunk])  # type: ignore[method-assign]
        crag_nodes.llm.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="refined ambiguous query"
        )

        app = build_crag_workflow(nodes=crag_nodes, settings=mock_settings)
        initial_state = create_initial_state("vague docker question")
        final_state = await app.ainvoke(initial_state)

        assert final_state["transformation_count"] == 1
        assert final_state["rewritten_query"] == "refined ambiguous query"
        assert final_state["relevance_status"] == "incorrect"
        assert final_state["citations"] == []
        assert final_state["refusal_reason"] == "insufficient_context"
        # LLM generate was only called once (for query rewrite, NOT for answer generation)
        assert crag_nodes.llm.generate.await_count == 1  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_workflow_incorrect_path_refuses_immediately(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        low_score_chunk: RetrievedChunk,
    ) -> None:
        """Low-relevance chunks (< 0.35) route directly to refuse without rewrite or LLM call."""
        crag_nodes.dense.asearch = AsyncMock(return_value=[low_score_chunk])  # type: ignore[method-assign]
        crag_nodes.sparse.asearch = AsyncMock(return_value=[low_score_chunk])  # type: ignore[method-assign]
        crag_nodes.ranker.arerank = AsyncMock(return_value=[low_score_chunk])  # type: ignore[method-assign]
        crag_nodes.llm.generate = AsyncMock()  # type: ignore[method-assign]

        app = build_crag_workflow(nodes=crag_nodes, settings=mock_settings)
        initial_state = create_initial_state("چگونه پیتزا بپزیم؟")
        final_state = await app.ainvoke(initial_state)

        assert final_state["transformation_count"] == 0
        assert final_state["relevance_status"] == "incorrect"
        assert final_state["citations"] == []
        assert "مستندات" in final_state["generation"] or "موجود نیست" in final_state["generation"]
        crag_nodes.llm.generate.assert_not_awaited()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_workflow_compiles_with_checkpointer_and_persists_state(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        high_score_chunk: RetrievedChunk,
    ) -> None:
        """Compiling with MemorySaver checkpointer persists state by thread_id."""
        crag_nodes.dense.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.sparse.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.ranker.arerank = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.llm.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="Grounded answer from checkpointed workflow."
        )

        memory = MemorySaver()
        app = build_crag_workflow(
            nodes=crag_nodes,
            settings=mock_settings,
            checkpointer=memory,
        )

        config: dict[str, Any] = {"configurable": {"thread_id": "thread-test-123"}}
        initial_state = create_initial_state("How to configure IPC?", trace_id="trace-123")
        final_state = await app.ainvoke(initial_state, config=config)  # type: ignore[arg-type]

        saved_snapshot = await app.aget_state(config)  # type: ignore[arg-type]
        assert saved_snapshot.values["trace_id"] == "trace-123"
        assert saved_snapshot.values["generation"] == final_state["generation"]

    @pytest.mark.asyncio
    async def test_crag_workflow_facade_ainvoke_and_astream(
        self,
        crag_nodes: CRAGNodes,
        mock_settings: Settings,
        high_score_chunk: RetrievedChunk,
    ) -> None:
        """CRAGWorkflow wrapper executes ainvoke and astream with checkpointer=True."""
        crag_nodes.dense.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.sparse.asearch = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.ranker.arerank = AsyncMock(return_value=[high_score_chunk])  # type: ignore[method-assign]
        crag_nodes.llm.generate = AsyncMock(  # type: ignore[method-assign]
            return_value="Streamed grounded response."
        )

        workflow = CRAGWorkflow(
            nodes=crag_nodes,
            settings=mock_settings,
            checkpointer=True,
        )

        result = await workflow.ainvoke(
            query="How to configure Docker IPC?",
            trace_id="trace-facade-1",
            thread_id="thread-facade-1",
        )
        assert result["generation"] == "Streamed grounded response."

        observed_steps: list[str] = []
        async for step_event in workflow.astream(
            query="How to configure Docker IPC?",
            trace_id="trace-facade-2",
            thread_id="thread-facade-2",
        ):
            observed_steps.extend(step_event.keys())

        assert "retrieve" in observed_steps
        assert "grade_documents" in observed_steps
        assert "generate" in observed_steps
