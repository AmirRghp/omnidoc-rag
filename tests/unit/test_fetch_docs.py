"""Unit tests for technical documentation fetcher and seeder."""

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.fetch_docs import (
    OFFLINE_FALLBACK_DOCS,
    fetch_from_github_api,
    populate_documentation,
    seed_offline_documentation,
)


def test_seed_offline_documentation_creates_all_files(tmp_path: Path) -> None:
    """Verify that offline seeding generates all predefined markdown reference files."""
    seeded = seed_offline_documentation(tmp_path)

    assert len(seeded) == len(OFFLINE_FALLBACK_DOCS)
    for file_path in seeded:
        assert file_path.exists()
        assert file_path.suffix == ".md"
        content = file_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0
        assert content.startswith("# ")


def test_seed_offline_documentation_idempotent(tmp_path: Path) -> None:
    """Verify that multiple seeding runs cleanly overwrite without duplication or error."""
    first_run = seed_offline_documentation(tmp_path)
    second_run = seed_offline_documentation(tmp_path)

    assert len(first_run) == len(second_run)
    assert len(list(tmp_path.glob("*.md"))) == len(OFFLINE_FALLBACK_DOCS)


def test_populate_documentation_force_offline(tmp_path: Path) -> None:
    """Verify that force_offline flag immediately executes seed without network calls."""
    with patch("scripts.fetch_docs.fetch_from_github_api") as mock_fetch:
        files = populate_documentation(target_dir=tmp_path, force_offline=True)
        mock_fetch.assert_not_called()
        assert len(files) == len(OFFLINE_FALLBACK_DOCS)
        assert (tmp_path / "docker_overview.md").exists()


def test_populate_documentation_fallback_on_network_error(tmp_path: Path) -> None:
    """Verify automatic fallback to offline seeding when GitHub API raises URLError."""
    with patch(
        "scripts.fetch_docs.fetch_from_github_api",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        files = populate_documentation(target_dir=tmp_path, force_offline=False)
        assert len(files) >= len(OFFLINE_FALLBACK_DOCS)
        assert (tmp_path / "postgres_overview.md").exists()


def test_fetch_from_github_api_mocked_success(tmp_path: Path) -> None:
    """Verify that fetch_from_github_api parses API response and writes markdown files."""
    mock_tree = [
        {
            "name": "install.md",
            "type": "file",
            "download_url": "https://raw.githubusercontent.com/docker/docs/main/install.md",
        },
        {
            "name": "image.png",
            "type": "file",
            "download_url": "https://raw.githubusercontent.com/docker/docs/main/image.png",
        },
    ]

    mock_api_resp = MagicMock()
    mock_api_resp.read.return_value = json.dumps(mock_tree).encode("utf-8")
    mock_api_resp.__enter__.return_value = mock_api_resp

    mock_file_resp = MagicMock()
    mock_file_resp.read.return_value = b"# Installing Docker Engine\nRun apt install."
    mock_file_resp.__enter__.return_value = mock_file_resp

    with patch("urllib.request.urlopen", side_effect=[mock_api_resp, mock_file_resp]):
        saved = fetch_from_github_api(
            repo="docker/docs",
            path="content/manuals",
            target_dir=tmp_path,
            max_files=10,
        )

    assert len(saved) == 1
    assert saved[0].name == "install.md"
    assert saved[0].read_text(encoding="utf-8") == "# Installing Docker Engine\nRun apt install."


def test_sample_persian_docker_guide_content_integrity() -> None:
    """Verify the sample Persian technical guide exists and contains required technical tokens."""
    guide_path = Path("data/raw/sample_persian_docker_guide.md")
    assert guide_path.exists(), "sample_persian_docker_guide.md must exist in data/raw/"

    content = guide_path.read_text(encoding="utf-8")
    # Verify critical technical tokens from PRD / Flow 4
    assert "ERR_DB_TIMEOUT" in content
    assert "502 Bad Gateway" in content
    assert "--ipc=host" in content
    assert "permission denied" in content
    assert "shared_buffers" in content
    assert "max_connections" in content
    # Verify Persian content presence
    assert "داکر" in content
    assert "پایگاه داده" in content
