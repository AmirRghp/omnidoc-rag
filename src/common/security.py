"""Bilingual input sanitization and Persian/English Unicode security normalization."""

import re

from src.common.exceptions import InvalidInputSecurityError

MAX_QUERY_LENGTH: int = 1000

# Translation table mapping Arabic Yeh (ي), Alef Maksura (ى), and Kaf (ك) to Persian (ی, ک)
_PERSIAN_CHAR_MAP: dict[int, int] = str.maketrans(
    "\u064a\u0649\u0643",
    "\u06cc\u06cc\u06a9",
)

# Regex matching invisible bidirectional overrides, directional marks, and null bytes.
# Preserves U+200C (Zero-Width Non-Joiner / نیم‌فاصله) required by Persian orthography.
_DIRECTIONAL_AND_CONTROL_PATTERN: re.Pattern[str] = re.compile(
    r"[\x00\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]"
)


def normalize_persian_text(text: str) -> str:
    """Normalize Arabic character variants (ي, ى, ك) to standard Persian (ی, ک).

    Preserves Persian Zero-Width Non-Joiner (U+200C) and English technical tokens.
    """
    return text.translate(_PERSIAN_CHAR_MAP)


def strip_directional_overrides(text: str) -> str:
    """Strip invisible Unicode directional override characters and null bytes."""
    return _DIRECTIONAL_AND_CONTROL_PATTERN.sub("", text)


def sanitize_query(query: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize, normalize, and validate a bilingual user query.

    Args:
        query: Raw user query string.
        max_length: Maximum allowed character length (defaults to 1,000).

    Returns:
        Cleaned, normalized, and stripped query string.

    Raises:
        InvalidInputSecurityError: If query is not a string, is empty after
            stripping invisible/whitespace characters, or exceeds max_length.
    """
    if not isinstance(query, str):
        raise InvalidInputSecurityError(
            message="Query must be a valid string",
            detail="Input rejected: expected a string value.",
        )

    cleaned = strip_directional_overrides(query)
    normalized = normalize_persian_text(cleaned).strip()

    if not normalized:
        raise InvalidInputSecurityError(
            message="Query cannot be empty",
            detail="Input rejected: query is empty or contains only whitespace/control characters.",
        )

    if len(normalized) > max_length:
        raise InvalidInputSecurityError(
            message=f"Query exceeds maximum length of {max_length} characters",
            detail=(
                f"Input rejected: query length ({len(normalized)}) exceeds maximum allowed "
                f"limit of {max_length} characters."
            ),
        )

    return normalized
