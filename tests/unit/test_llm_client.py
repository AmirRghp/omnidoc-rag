"""Unit tests for local vLLM OpenAI-compatible client (Task 6.2)."""

import json

import httpx
import pytest

from src.common.config import Settings
from src.common.exceptions import InferenceEngineUnavailableError
from src.services.llm_client import ChatMessage, LLMClient, LLMResponse, VLLMClient


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing isolated mock settings for testing."""
    return Settings(
        LLM_BASE_URL="http://mock-vllm:8000/v1",
        LLM_API_KEY="mock-test-key",
        LLM_MODEL="Qwen/Qwen2.5-7B-Instruct",
    )


class TestVLLMClientInitialization:
    """Tests for client initialization, URL resolution, and lifecycle."""

    def test_default_initialization_from_settings(self, mock_settings: Settings) -> None:
        """Client properly adopts configuration from settings."""
        client = VLLMClient(settings=mock_settings)

        assert client.base_url == "http://mock-vllm:8000/v1"
        assert client.api_key == "mock-test-key"
        assert client.model == "Qwen/Qwen2.5-7B-Instruct"
        assert client.chat_completions_url == "http://mock-vllm:8000/v1/chat/completions"
        assert client.models_url == "http://mock-vllm:8000/v1/models"

    def test_url_construction_without_v1_suffix(self) -> None:
        """Client automatically appends /v1 when missing from base_url."""
        client = VLLMClient(base_url="http://localhost:8000")

        assert client.chat_completions_url == "http://localhost:8000/v1/chat/completions"
        assert client.models_url == "http://localhost:8000/v1/models"

    def test_url_construction_with_trailing_slash(self) -> None:
        """Client strips trailing slashes gracefully."""
        client = VLLMClient(base_url="http://localhost:8000/v1/")

        assert client.chat_completions_url == "http://localhost:8000/v1/chat/completions"
        assert client.models_url == "http://localhost:8000/v1/models"

    def test_convenience_alias(self) -> None:
        """LLMClient is an alias of VLLMClient."""
        assert LLMClient is VLLMClient

    @pytest.mark.asyncio
    async def test_async_context_manager_lifecycle(self) -> None:
        """Client properly closes internal session on context exit."""
        async with VLLMClient(base_url="http://mock-vllm:8000/v1") as client:
            assert not client._client.is_closed
        assert client._client.is_closed


class TestVLLMClientGeneration:
    """Tests for non-streaming text generation."""

    @pytest.mark.asyncio
    async def test_generate_string_success(self, mock_settings: Settings) -> None:
        """generate() executes chat completion and returns string content."""
        recorded_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            recorded_requests.append(request)
            body = json.loads(request.read())
            assert body["model"] == "Qwen/Qwen2.5-7B-Instruct"
            expected_msgs = [{"role": "user", "content": "How do I configure Docker IPC?"}]
            assert body["messages"] == expected_msgs
            assert body["stream"] is False

            response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Use `--ipc=host` in your docker run command.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 10,
                    "total_tokens": 25,
                },
            }
            return httpx.Response(200, json=response_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            result = await client.generate("How do I configure Docker IPC?")

        assert result == "Use `--ipc=host` in your docker run command."
        assert len(recorded_requests) == 1
        assert recorded_requests[0].headers["Authorization"] == "Bearer mock-test-key"

    @pytest.mark.asyncio
    async def test_generate_complete_returns_structured_llm_response(
        self, mock_settings: Settings
    ) -> None:
        """generate_complete() returns complete LLMResponse with token usage metrics."""

        def handler(request: httpx.Request) -> httpx.Response:
            response_data = {
                "id": "chatcmpl-456",
                "object": "chat.completion",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "PostgreSQL shared_buffers setup.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 12,
                    "total_tokens": 42,
                },
            }
            return httpx.Response(200, json=response_data)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            response = await client.generate_complete(
                [
                    ChatMessage(role="system", content="You are a DBA."),
                    ChatMessage(role="user", content="Explain shared_buffers."),
                ],
                temperature=0.2,
                max_tokens=256,
            )

        assert isinstance(response, LLMResponse)
        assert response.content == "PostgreSQL shared_buffers setup."
        assert response.prompt_tokens == 30
        assert response.completion_tokens == 12
        assert response.total_tokens == 42
        assert response.finish_reason == "stop"


class TestVLLMClientStreaming:
    """Tests for token-by-token streaming via SSE."""

    @pytest.mark.asyncio
    async def test_stream_generate_yields_tokens_sequentially(
        self, mock_settings: Settings
    ) -> None:
        """stream_generate() yields token chunks until [DONE] signal."""
        sse_lines = [
            'data: {"id":"1","choices":[{"index":0,"delta":{"content":"برای"}}]}\n\n',
            'data: {"id":"1","choices":[{"index":0,"delta":{"content":" اجرای"}}]}\n\n',
            'data: {"id":"1","choices":[{"index":0,"delta":{"content":" داکر"}}]}\n\n',
            ": keep-alive comment\n\n",
            'data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        sse_bytes = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            assert body["stream"] is True
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_bytes,
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            tokens: list[str] = []
            async for token in client.stream_generate("دستور اجرای داکر چیست؟"):
                tokens.append(token)

        assert tokens == ["برای", " اجرای", " داکر"]

    @pytest.mark.asyncio
    async def test_stream_generate_ignores_malformed_json_lines(
        self, mock_settings: Settings
    ) -> None:
        """Stream parser safely ignores invalid JSON chunks without throwing."""
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            "data: {invalid-json-content}\n\n",
            'data: {"choices":[{"delta":{"content":" World"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        sse_bytes = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_bytes)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            tokens = [t async for t in client.stream_generate("Hi")]

        assert tokens == ["Hello", " World"]


class TestVLLMClientRetriesAndErrorHandling:
    """Tests for exponential backoff, retry mechanisms, and domain exceptions."""

    @pytest.mark.asyncio
    async def test_transient_503_recovers_on_retry(self, mock_settings: Settings) -> None:
        """Client retries on transient 503 response and succeeds on subsequent attempt."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(503, text="Service Unavailable / Model Loading")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Successfully recovered!"}}],
                    "usage": {"total_tokens": 10},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(
                client=http_client,
                settings=mock_settings,
                max_retries=3,
                retry_backoff_factor=0.01,
            )
            result = await client.generate("Ping")

        assert result == "Successfully recovered!"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_inference_engine_unavailable_error(
        self, mock_settings: Settings
    ) -> None:
        """Client raises InferenceEngineUnavailableError when max retries are exceeded."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(503, text="GPU OOM")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(
                client=http_client,
                settings=mock_settings,
                max_retries=2,
                retry_backoff_factor=0.01,
            )
            with pytest.raises(InferenceEngineUnavailableError) as exc_info:
                await client.generate("Ping")

        assert call_count == 3  # Initial + 2 retries
        assert exc_info.value.status_code == 503
        assert exc_info.value.code == "ERR_VLLM_CONNECTION_FAILED"

    @pytest.mark.asyncio
    async def test_connection_error_retries_and_raises(self, mock_settings: Settings) -> None:
        """Network ConnectError triggers retries and wraps in domain exception."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("Connection refused by peer")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(
                client=http_client,
                settings=mock_settings,
                max_retries=2,
                retry_backoff_factor=0.01,
            )
            with pytest.raises(InferenceEngineUnavailableError) as exc_info:
                await client.generate("Ping")

        assert call_count == 3
        assert "Connection refused" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_client_error_fails_immediately_without_retries(
        self, mock_settings: Settings
    ) -> None:
        """HTTP 400 Bad Request raises domain exception without wasted retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, text="Bad Request: Unknown parameter")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(
                client=http_client,
                settings=mock_settings,
                max_retries=3,
                retry_backoff_factor=0.01,
            )
            with pytest.raises(InferenceEngineUnavailableError):
                await client.generate("Ping")

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_streaming_transient_error_recovers_on_retry(
        self, mock_settings: Settings
    ) -> None:
        """Stream connection recovers on initial retry and streams tokens successfully."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(502, text="Bad Gateway")
            sse_content = 'data: {"choices":[{"delta":{"content":"Ready"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, content=sse_content.encode("utf-8"))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(
                client=http_client,
                settings=mock_settings,
                max_retries=2,
                retry_backoff_factor=0.01,
            )
            tokens = [t async for t in client.stream_generate("Test")]

        assert tokens == ["Ready"]
        assert call_count == 2


class TestVLLMClientHealthCheck:
    """Tests for health check probe."""

    @pytest.mark.asyncio
    async def test_check_health_returns_true_on_health_endpoint_200(
        self, mock_settings: Settings
    ) -> None:
        """check_health returns True when /health responds with 200."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health":
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            assert await client.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_returns_true_on_models_endpoint_fallback(
        self, mock_settings: Settings
    ) -> None:
        """check_health falls back to /models when /health returns 404."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/health":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "Qwen"}]})
            return httpx.Response(500)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            assert await client.check_health() is True

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_total_failure(
        self, mock_settings: Settings
    ) -> None:
        """check_health returns False when server is unreachable or failing."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Down")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http_client:
            client = VLLMClient(client=http_client, settings=mock_settings)
            assert await client.check_health() is False
