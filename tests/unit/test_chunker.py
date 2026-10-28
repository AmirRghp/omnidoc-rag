"""Unit tests for Header-Aware and Code-Safe MarkdownSemanticChunker."""

from pathlib import Path

import pytest

from src.common.exceptions import DocumentParsingError
from src.common.schemas import generate_point_id
from src.ingestion.chunker import MarkdownSemanticChunker


class TestHeaderSplittingAndMetadata:
    """Tests for Markdown header detection and section_heading attachment."""

    def test_splits_by_h1_h2_h3_headers(self) -> None:
        """Chunker must split sections on #, ##, and ### headers and preserve headings."""
        markdown = (
            "# Docker Engine Overview\n"
            "Docker Engine is an open source containerization technology.\n\n"
            "## Daemon Configuration\n"
            "Configure the Docker daemon using `/etc/docker/daemon.json`.\n\n"
            "### Socket Options\n"
            "Use `-H unix:///var/run/docker.sock` to bind to the local Unix socket.\n"
        )
        chunker = MarkdownSemanticChunker(max_tokens=512)
        chunks = chunker.chunk_text(markdown, source_file="manuals/docker_overview.md")

        assert len(chunks) == 3
        assert chunks[0].section_heading == "# Docker Engine Overview"
        assert "containerization technology" in chunks[0].content
        assert chunks[0].chunk_id == 0
        assert chunks[0].point_id == generate_point_id("manuals/docker_overview.md", 0)

        assert chunks[1].section_heading == "## Daemon Configuration"
        assert "daemon.json" in chunks[1].content
        assert chunks[1].chunk_id == 1

        assert chunks[2].section_heading == "### Socket Options"
        assert "docker.sock" in chunks[2].content
        assert chunks[2].chunk_id == 2

    def test_content_before_first_header_uses_default_heading(self) -> None:
        """Content appearing before any header should use '# Introduction'."""
        markdown = (
            "Preliminary notes on PostgreSQL memory tuning.\n\n"
            "## Shared Buffers\n"
            "Set `shared_buffers` to 25% of system RAM.\n"
        )
        chunker = MarkdownSemanticChunker(max_tokens=512)
        chunks = chunker.chunk_text(markdown, source_file="manuals/postgres.md")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "# Introduction"
        assert "Preliminary notes" in chunks[0].content
        assert chunks[1].section_heading == "## Shared Buffers"

    def test_yaml_frontmatter_title_is_extracted_and_stripped(self) -> None:
        """Hugo/YAML frontmatter should be stripped and its title used for intro text."""
        markdown = (
            "---\n"
            "title: Install Docker Engine on Ubuntu\n"
            "weight: 10\n"
            "---\n\n"
            "To get started with Docker Engine on Ubuntu, meet the prerequisites.\n\n"
            "## Prerequisites\n"
            "You need a 64-bit Ubuntu version.\n"
        )
        chunker = MarkdownSemanticChunker(max_tokens=512)
        chunks = chunker.chunk_text(markdown, source_file="manuals/ubuntu.md")

        assert len(chunks) == 2
        assert chunks[0].section_heading == "# Install Docker Engine on Ubuntu"
        assert "weight: 10" not in chunks[0].content
        assert chunks[1].section_heading == "## Prerequisites"


class TestCodeBlockAndTableIntegrity:
    """Tests ensuring code blocks and Markdown tables are never broken mid-structure."""

    def test_fenced_code_block_with_hash_comments_never_splits(self) -> None:
        """Comments starting with '#' inside fenced code blocks must not trigger header splits."""
        code_block = (
            "```bash\n"
            "# Create docker group if missing\n"
            "sudo groupadd docker\n\n"
            "## Add current user to docker group\n"
            "sudo usermod -aG docker $USER\n\n"
            "### Activate group membership\n"
            "newgrp docker\n"
            "```"
        )
        markdown = (
            "## Fixing Docker Socket Permission Denied\n"
            "Run the following commands to resolve the socket error:\n\n"
            f"{code_block}\n\n"
            "After running these commands, verify with `docker ps`.\n"
        )
        chunker = MarkdownSemanticChunker(max_tokens=40)
        chunks = chunker.chunk_text(markdown, source_file="manuals/docker_permissions.md")

        assert all(c.section_heading == "## Fixing Docker Socket Permission Denied" for c in chunks)
        matching_code_chunks = [c for c in chunks if "```bash" in c.content]
        assert len(matching_code_chunks) == 1
        assert code_block in matching_code_chunks[0].content

    def test_markdown_table_never_splits_mid_table(self) -> None:
        """Markdown tables (| col1 | col2 |) must remain intact inside a single chunk."""
        table_block = (
            "| Parameter | Diagnostic Command | Expected State |\n"
            "|---|---|---|\n"
            "| Container Status | `docker ps -a --filter status=running` | `Up X minutes` |\n"
            "| Shared Network | `docker network inspect app_network` | Both container IPs |\n"
            "| Listening Port | `netstat -tuln` | Bound to `0.0.0.0` |\n"
            "| Healthcheck | `docker inspect --format='{{json .State.Health}}'` | `healthy` |"
        )
        markdown = (
            "## Network Troubleshooting Checklist\n"
            "Introductory explanation before the diagnostic table.\n\n"
            f"{table_block}\n\n"
            "Concluding remarks after the table.\n"
        )
        # Set small max_tokens so prose and table cannot fit together, forcing table into own chunk
        chunker = MarkdownSemanticChunker(max_tokens=30)
        chunks = chunker.chunk_text(markdown, source_file="manuals/network.md")

        table_chunks = [c for c in chunks if "| Parameter | Diagnostic Command |" in c.content]
        assert len(table_chunks) == 1
        assert table_block in table_chunks[0].content
        assert table_chunks[0].content.count("\n") >= 5


class TestLanguageDetectionAndTokenCount:
    """Tests for bilingual language detection ('fa' vs 'en') and tiktoken token counting."""

    def test_detects_english_and_persian_accurately(self) -> None:
        """Language detector must distinguish English from Persian even with technical flags."""
        chunker = MarkdownSemanticChunker()

        en_text = "Use `--ipc=host` or `--shm-size=8g` when running PyTorch containers."
        fa_text = (
            "برای رفع خطای حافظه اشتراکی در کانتینرهای Docker از فلگ `--ipc=host` استفاده کنید."
        )
        fa_with_large_code = (
            "برای پیکربندی شبکه در Docker Compose فایل زیر را ایجاد کنید:\n\n"
            "```yaml\n"
            "version: '3.8'\n"
            "services:\n"
            "  backend-api:\n"
            "    image: python:3.11-slim\n"
            "    command: uvicorn main:app --host 0.0.0.0 --port 8000\n"
            "```"
        )

        assert chunker.detect_language(en_text) == "en"
        assert chunker.detect_language(fa_text) == "fa"
        assert chunker.detect_language(fa_with_large_code) == "fa"

    def test_token_count_is_positive_and_consistent(self) -> None:
        """Token count must be positive for non-empty strings and zero for empty strings."""
        chunker = MarkdownSemanticChunker()
        assert chunker.count_tokens("") == 0
        en_tokens = chunker.count_tokens("Docker container networking bridge driver")
        fa_tokens = chunker.count_tokens("راهنمای جامع عیب‌یابی و پیکربندی پیشرفته داکر")
        assert en_tokens >= 5
        assert fa_tokens >= 5

    def test_oversized_prose_splits_cleanly_within_max_tokens(self) -> None:
        """Long prose paragraphs without headers should be split to respect max_tokens."""
        sentences = [
            f"Sentence number {i} explains PostgreSQL connection pooling and lock contention."
            for i in range(20)
        ]
        markdown = "## Connection Pooling\n" + " ".join(sentences)
        chunker = MarkdownSemanticChunker(max_tokens=50)
        chunks = chunker.chunk_text(markdown, source_file="manuals/pooling.md")

        assert len(chunks) > 3
        for chunk in chunks:
            assert chunk.section_heading == "## Connection Pooling"
            assert chunk.token_count <= 50

    def test_chunk_sample_persian_docker_guide_file(self) -> None:
        """Chunking the real sample_persian_docker_guide.md must preserve structure and 'fa'."""
        guide_path = Path("data/raw/sample_persian_docker_guide.md")
        chunker = MarkdownSemanticChunker(max_tokens=512)
        chunks = chunker.chunk_file(guide_path, root_dir=Path("data/raw"))

        assert len(chunks) >= 5
        assert all(c.source_file == "sample_persian_docker_guide.md" for c in chunks)
        assert all(c.language == "fa" for c in chunks)
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

        # Verify code blocks in chunks have balanced triple backticks
        for chunk in chunks:
            fence_count = chunk.content.count("```")
            assert fence_count % 2 == 0, f"Unbalanced code fence in chunk {chunk.chunk_id}"


class TestChunkerErrorHandling:
    """Tests for error handling and input validation in MarkdownSemanticChunker."""

    def test_invalid_max_tokens_raises_value_error(self) -> None:
        """Non-positive max_tokens must raise ValueError at initialization."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            MarkdownSemanticChunker(max_tokens=0)

    def test_empty_or_whitespace_text_raises_document_parsing_error(self) -> None:
        """Empty or whitespace-only text must raise DocumentParsingError."""
        chunker = MarkdownSemanticChunker()
        with pytest.raises(DocumentParsingError):
            chunker.chunk_text("   \n\t  ", source_file="empty.md")

    def test_missing_file_raises_document_parsing_error(self) -> None:
        """Attempting to chunk a non-existent file must raise DocumentParsingError."""
        chunker = MarkdownSemanticChunker()
        with pytest.raises(DocumentParsingError):
            chunker.chunk_file("data/raw/non_existent_file_12345.md")
