"""Asynchronous OpenAI-compatible client for local vLLM inference."""

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from src.common.config import Settings, create_settings
from src.common.exceptions import InferenceEngineUnavailableError
from src.common.logger import get_logger

logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """Chat message schema for OpenAI-compatible endpoints."""

    model_config = ConfigDict(strict=True, frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """Complete response payload returned by LLM generation."""

    model_config = ConfigDict(strict=True, frozen=True)

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None


class VLLMClient:
    """Async OpenAI-compatible client for local vLLM inference with retries and streaming."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the vLLM client.

        Args:
            base_url: Base URL for vLLM API (e.g. http://localhost:8000/v1).
            api_key: API authorization key.
            model: Model identifier.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum retry attempts for transient connection/server errors.
            retry_backoff_factor: Multiplier for exponential backoff between retries.
            client: Optional injected httpx.AsyncClient instance for connection reuse.
            settings: Optional application settings.
        """
        self.settings = settings or create_settings()
        self.base_url = (base_url or self.settings.llm_base_url).rstrip("/")
        self.api_key = api_key or self.settings.llm_api_key
        self.model = model or self.settings.llm_model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.retry_backoff_factor = max(0.0, retry_backoff_factor)

        self._injected_client = client is not None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=5.0),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def chat_completions_url(self) -> str:
        """Construct the full URL for chat completions endpoint."""
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @property
    def models_url(self) -> str:
        """Construct the full URL for models endpoint."""
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/models"
        return f"{self.base_url}/v1/models"

    def _format_messages(
        self,
        messages: Sequence[ChatMessage | dict[str, str]] | str,
    ) -> list[dict[str, str]]:
        """Normalize messages input into OpenAI wire-format list of dictionaries."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        formatted: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted.append({"role": msg.role, "content": msg.content})
            else:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                formatted.append({"role": role, "content": content})
        return formatted

    def _build_payload(
        self,
        messages: Sequence[ChatMessage | dict[str, str]] | str,
        *,
        stream: bool,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build request JSON payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": stream,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if extra_params:
            payload.update(extra_params)
        return payload

    def _get_headers(self) -> dict[str, str]:
        """Build request headers including bearer authentication."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate(
        self,
        messages: Sequence[ChatMessage | dict[str, str]] | str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """Execute complete text generation and return content string.

        Args:
            messages: Prompt string, list of message dicts, or ChatMessage instances.
            temperature: Sampling temperature.
            max_tokens: Optional token generation budget.
            stop: Optional stop sequences.
            extra_params: Optional additional parameters forwarded to the model.

        Returns:
            The generated assistant text response.
        """
        response_model = await self.generate_complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_params=extra_params,
        )
        return response_model.content

    async def generate_complete(
        self,
        messages: Sequence[ChatMessage | dict[str, str]] | str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Execute complete text generation and return structured LLMResponse model.

        Args:
            messages: Prompt string, list of message dicts, or ChatMessage instances.
            temperature: Sampling temperature.
            max_tokens: Optional token generation budget.
            stop: Optional stop sequences.
            extra_params: Optional additional parameters forwarded to the model.

        Returns:
            LLMResponse containing generated content, model name, and token usage counts.
        """
        payload = self._build_payload(
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_params=extra_params,
        )

        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(
                    self.chat_completions_url,
                    json=payload,
                    headers=self._get_headers(),
                )
                if resp.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                    backoff = self.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "vLLM server returned HTTP %d (attempt %d/%d). Retrying in %.2fs...",
                        resp.status_code,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                if resp.is_error:
                    raise InferenceEngineUnavailableError(
                        message="vLLM server returned an error response",
                        detail=(
                            f"HTTP {resp.status_code} from {self.chat_completions_url}: "
                            f"{resp.text[:500]}"
                        ),
                    )

                data: dict[str, Any] = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return LLMResponse(content="", model=self.model)

                first_choice = choices[0]
                message_dict = first_choice.get("message", {})
                content = message_dict.get("content", "") or ""
                finish_reason = first_choice.get("finish_reason")

                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    model=str(data.get("model", self.model)),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                    finish_reason=finish_reason,
                )

            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "Connection error communicating with vLLM (attempt %d/%d): %s. "
                        "Retrying in %.2fs...",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                raise InferenceEngineUnavailableError(
                    message="Failed to connect to local vLLM inference server",
                    detail=(
                        f"Connection to {self.chat_completions_url} failed after "
                        f"{self.max_retries + 1} attempts: {exc}"
                    ),
                ) from exc

        raise InferenceEngineUnavailableError(
            message="vLLM inference request failed after retries",
            detail=(
                f"Exceeded max retries ({self.max_retries}) connecting to "
                f"{self.chat_completions_url}"
            ),
        )

    async def stream_generate(
        self,
        messages: Sequence[ChatMessage | dict[str, str]] | str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens incrementally from vLLM via Server-Sent Events (SSE).

        Args:
            messages: Prompt string, list of message dicts, or ChatMessage instances.
            temperature: Sampling temperature.
            max_tokens: Optional token generation budget.
            stop: Optional stop sequences.
            extra_params: Optional additional parameters forwarded to the model.

        Yields:
            Token chunks as received from the model.
        """
        payload = self._build_payload(
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            extra_params=extra_params,
        )

        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = self._client.build_request(
                    "POST",
                    self.chat_completions_url,
                    json=payload,
                    headers=self._get_headers(),
                )
                resp = await self._client.send(req, stream=True)

                if resp.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                    await resp.aclose()
                    backoff = self.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "vLLM server returned HTTP %d for stream (attempt %d/%d). "
                        "Retrying in %.2fs...",
                        resp.status_code,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                if resp.is_error:
                    body = await resp.aread()
                    await resp.aclose()
                    raise InferenceEngineUnavailableError(
                        message="vLLM streaming request failed",
                        detail=(
                            f"HTTP {resp.status_code} from {self.chat_completions_url}: "
                            f"{body.decode('utf-8', errors='replace')[:500]}"
                        ),
                    )

                response = resp
                break

            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_factor * (2**attempt)
                    logger.warning(
                        "Connection error initiating stream with vLLM (attempt %d/%d): %s. "
                        "Retrying in %.2fs...",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    continue

                raise InferenceEngineUnavailableError(
                    message="Failed to connect to local vLLM streaming server",
                    detail=(
                        f"Stream connection to {self.chat_completions_url} failed after "
                        f"{self.max_retries + 1} attempts: {exc}"
                    ),
                ) from exc

        if response is None:
            raise InferenceEngineUnavailableError(
                message="Failed to initiate vLLM stream",
                detail=(
                    f"Exceeded max retries ({self.max_retries}) connecting to "
                    f"{self.chat_completions_url}"
                ),
            )

        try:
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue

                if line.startswith("data: "):
                    line = line[6:].strip()
                elif line.startswith("data:"):
                    line = line[5:].strip()

                if line == "[DONE]":
                    break

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content_chunk = delta.get("content")
                if content_chunk:
                    yield content_chunk

        except (httpx.HTTPError, Exception) as exc:
            if isinstance(exc, InferenceEngineUnavailableError):
                raise
            raise InferenceEngineUnavailableError(
                message="vLLM stream interrupted",
                detail=f"Stream from {self.chat_completions_url} was disconnected: {exc}",
            ) from exc
        finally:
            await response.aclose()

    async def check_health(self) -> bool:
        """Check if the vLLM server is healthy and responding.

        Tests the `/health` endpoint first, then falls back to `/models`.

        Returns:
            True if the server is healthy and returns 200, False otherwise.
        """
        health_url = f"{self.base_url}/health"
        candidate_urls = [health_url, self.models_url]

        for url in candidate_urls:
            try:
                resp = await self._client.get(
                    url,
                    headers=self._get_headers(),
                    timeout=httpx.Timeout(2.0, connect=2.0),
                )
                if resp.status_code == 200:
                    return True
            except (httpx.HTTPError, Exception):
                continue
        return False

    async def aclose(self) -> None:
        """Close the underlying HTTP client if not externally injected."""
        if not self._injected_client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "VLLMClient":
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context manager, closing client if internally owned."""
        await self.aclose()


# Convenience alias matching architectural references
LLMClient = VLLMClient
