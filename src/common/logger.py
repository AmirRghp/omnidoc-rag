"""Structured JSON logging with asynchronous trace_id context propagation."""

import contextlib
import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

# Context variable holding the correlation / trace ID for current execution thread/task
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


def set_trace_id(trace_id: str) -> None:
    """Store trace_id in the current asynchronous context."""
    trace_id_var.set(trace_id)


def get_trace_id() -> str | None:
    """Retrieve trace_id from the current asynchronous context."""
    return trace_id_var.get()


def clear_trace_id() -> None:
    """Clear trace_id from the current asynchronous context."""
    trace_id_var.set(None)


def inject_trace_id(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor injecting current trace_id into event dictionary."""
    current_trace = get_trace_id()
    if current_trace is not None:
        event_dict["trace_id"] = current_trace
    return event_dict


def configure_logger(
    log_level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure standard library logging and structlog processors."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clean existing root handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        with contextlib.suppress(Exception):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Standard stream handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    root_logger.addHandler(handler)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        inject_trace_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: structlog.types.Processor
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str = "omnidoc-rag") -> Any:
    """Get a structured logger instance bound with module or component name."""
    return structlog.get_logger(name)
