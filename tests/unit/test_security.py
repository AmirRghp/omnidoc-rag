"""Unit tests for bilingual input sanitizer and security normalizer."""

import pytest

from src.common.exceptions import InvalidInputSecurityError
from src.common.security import (
    MAX_QUERY_LENGTH,
    normalize_persian_text,
    sanitize_query,
    strip_directional_overrides,
)


class TestNormalizePersianText:
    """Tests for Arabic-to-Persian character normalization and ZWNJ preservation."""

    def test_converts_arabic_yeh_and_kaf_to_persian(self) -> None:
        """Arabic Yeh (ي, ى) and Kaf (ك) must normalize to Persian ی and ک."""
        raw = "خطاي داكر در پورت‌هاي شبكه و دسترسي كليد"
        normalized = normalize_persian_text(raw)
        assert normalized == "خطای داکر در پورت‌های شبکه و دسترسی کلید"
        assert "\u064a" not in normalized
        assert "\u0649" not in normalized
        assert "\u0643" not in normalized

    def test_converts_alef_maksura_to_persian_yeh(self) -> None:
        """Arabic Alef Maksura (ى U+0649) must normalize to Persian Yeh (ی U+06CC)."""
        assert normalize_persian_text("موسى و عيسى") == "موسی و عیسی"

    def test_preserves_zwnj_half_space(self) -> None:
        """Persian zero-width non-joiner (U+200C) must remain intact."""
        text_with_zwnj = "کانتینرها، پورت‌ها و سرویس‌هایی که اجرا می‌شوند"
        normalized = normalize_persian_text(text_with_zwnj)
        assert "\u200c" in normalized
        assert normalized == text_with_zwnj

    def test_preserves_english_and_technical_tokens(self) -> None:
        """English technical flags, paths, and error codes must remain untouched."""
        raw = "Fix ERR_DB_TIMEOUT with --ipc=host in /var/run/docker.sock كد ۵۰۲"
        normalized = normalize_persian_text(raw)
        assert normalized == "Fix ERR_DB_TIMEOUT with --ipc=host in /var/run/docker.sock کد ۵۰۲"

    def test_applies_unicode_nfc_canonical_normalization(self) -> None:
        """Decomposed NFD characters are canonically composed via NFC."""
        decomposed = "e\u0301"  # 'e' + combining acute accent
        normalized = normalize_persian_text(decomposed)
        assert normalized == "\u00e9"



class TestStripDirectionalOverrides:
    """Tests for stripping bidirectional override and invisible control characters."""

    def test_strips_bidi_override_characters(self) -> None:
        """Characters U+202A..U+202E and U+2066..U+2069 must be removed."""
        spoofed = (
            "docker \u202erun \u202b--privileged \u202apostgres\u202c\u202d\u2066\u2067\u2068\u2069"
        )
        cleaned = strip_directional_overrides(spoofed)
        assert cleaned == "docker run --privileged postgres"
        for codepoint in (
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        ):
            assert codepoint not in cleaned

    def test_strips_directional_marks_and_null_bytes(self) -> None:
        """Directional marks (LRM, RLM, ALM) and null bytes must be stripped."""
        spoofed = "SELECT\x00 * FROM\u200e users\u200f WHERE\u061c id=1"
        cleaned = strip_directional_overrides(spoofed)
        assert cleaned == "SELECT * FROM users WHERE id=1"

    def test_preserves_standard_whitespace_and_zwnj(self) -> None:
        """Newlines, tabs, spaces, and Persian ZWNJ (U+200C) must not be stripped."""
        raw = "خط اول\nخط دوم\tبا نیم\u200cفاصله"
        assert strip_directional_overrides(raw) == raw


class TestSanitizeQuery:
    """Tests for end-to-end query sanitization, normalization, and validation."""

    def test_valid_bilingual_query_sanitized(self) -> None:
        """Valid mixed Persian/English query is stripped, cleaned, and normalized."""
        raw = "  \u202eچگونه خطاي permission denied در /var/run/docker.sock را رفع كنم؟\u202c  "
        sanitized = sanitize_query(raw)
        assert sanitized == "چگونه خطای permission denied در /var/run/docker.sock را رفع کنم؟"

    def test_exact_max_length_accepted(self) -> None:
        """Query with exact length of 1,000 characters must succeed."""
        exact_query = "a" * MAX_QUERY_LENGTH
        result = sanitize_query(exact_query)
        assert len(result) == MAX_QUERY_LENGTH
        assert result == exact_query

    def test_exceeding_max_length_raises_security_error(self) -> None:
        """Query exceeding 1,000 characters must raise InvalidInputSecurityError."""
        overlong_query = "الف" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(InvalidInputSecurityError) as exc_info:
            sanitize_query(overlong_query)

        err = exc_info.value
        assert err.code == "ERR_SECURITY_INPUT_REJECTED"
        assert err.status_code == 400
        assert "1000" in err.detail

    def test_custom_max_length_enforced(self) -> None:
        """Custom max_length parameter is respected when provided."""
        with pytest.raises(InvalidInputSecurityError):
            sanitize_query("short query", max_length=5)

    def test_empty_or_whitespace_only_raises_security_error(self) -> None:
        """Empty or whitespace-only queries must raise InvalidInputSecurityError."""
        for invalid in ("", "   ", "\n\t  \r"):
            with pytest.raises(InvalidInputSecurityError) as exc_info:
                sanitize_query(invalid)
            assert exc_info.value.code == "ERR_SECURITY_INPUT_REJECTED"
            assert exc_info.value.status_code == 400

    def test_only_invisible_overrides_raises_security_error(self) -> None:
        """Query containing only directional override characters becomes empty and fails."""
        with pytest.raises(InvalidInputSecurityError):
            sanitize_query("\u202e\u202b\u202c\u200e\u200f")

    def test_non_string_input_raises_security_error(self) -> None:
        """Non-string input must raise InvalidInputSecurityError."""
        with pytest.raises(InvalidInputSecurityError):
            sanitize_query(None)  # type: ignore[arg-type]
