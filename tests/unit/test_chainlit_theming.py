"""Unit tests for Chainlit configuration, bilingual RTL typography, and CSS theming (Task 9.1)."""

from pathlib import Path

import tomli
from chainlit.server import app as chainlit_app
from starlette.testclient import TestClient

import app


def test_chainlit_config_file() -> None:
    """Validate that .chainlit/config.toml exists and configures dark mode and custom CSS."""
    config_path = Path(".chainlit/config.toml")
    assert config_path.is_file(), ".chainlit/config.toml must exist"

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    ui = data.get("UI", {})
    assert ui.get("name") == "OmniDoc RAG"
    assert ui.get("default_theme") == "dark"
    assert ui.get("custom_css") == "/public/custom.css"

    features = data.get("features", {})
    assert features.get("unsafe_allow_html") is True


def test_custom_css_typography_and_rtl_rules() -> None:
    """Validate that public/custom.css defines required fonts and RTL isolation."""
    css_path = Path("public/custom.css")
    assert css_path.is_file(), "public/custom.css must exist"

    content = css_path.read_text(encoding="utf-8")
    assert "Vazirmatn" in content, "Vazirmatn Persian font must be imported and configured"
    assert "Inter" in content, "Inter English font must be configured"
    assert "JetBrains Mono" in content, "JetBrains Mono code font must be configured"

    # Strict LTR isolation for code
    assert "direction: ltr !important" in content, "Code blocks must enforce LTR direction"
    assert "unicode-bidi: isolate" in content, "Inline and block code must isolate bidi text"

    # Design tokens from DESIGN_SYSTEM.md
    assert "#090D16" in content, "Dark background color (#090D16) must be defined"
    assert "#131B2E" in content, "Surface container color (#131B2E) must be defined"
    assert "#10B981" in content, "Emerald primary color (#10B981) must be defined"


def test_chainlit_welcome_markdown() -> None:
    """Validate that chainlit.md contains bilingual introductory guidance."""
    md_path = Path("chainlit.md")
    assert md_path.is_file(), "chainlit.md must exist"

    content = md_path.read_text(encoding="utf-8")
    assert "OmniDoc RAG" in content
    assert "Docker" in content
    assert "PostgreSQL" in content
    assert "Linux" in content
    assert "داکر" in content


def test_chainlit_app_handlers_defined() -> None:
    """Validate that app.py defines the on_chat_start and on_message hooks."""
    assert hasattr(app, "on_chat_start"), "app.py must define on_chat_start callback"
    assert hasattr(app, "on_message"), "app.py must define on_message callback"


def test_chainlit_server_serves_custom_css() -> None:
    """Validate that the Chainlit Starlette server serves /public/custom.css with HTTP 200."""
    client = TestClient(chainlit_app)
    response = client.get("/public/custom.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "Vazirmatn" in response.text
