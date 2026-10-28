"""Automated verification suite for portfolio-grade README.md integrity."""

import re
from pathlib import Path


def test_readme_file_exists_and_non_empty() -> None:
    """Verify that README.md exists in repository root and has substantial content."""
    readme_path = Path("README.md")
    assert readme_path.is_file(), "README.md does not exist in repository root."
    content = readme_path.read_text(encoding="utf-8")
    assert len(content) > 2000, "README.md is too short to be portfolio-grade."


def test_readme_contains_all_core_portfolio_sections() -> None:
    """Verify that all core technical and architectural sections exist in README.md."""
    content = Path("README.md").read_text(encoding="utf-8")

    expected_sections = [
        "OmniDoc RAG",
        'The "Demo vs. Production" Gap',
        "System Architecture",
        "Quantitative Benchmark Results",
        "Architectural Decision Records",
        "Bilingual Persian/English UX",
        "One-Command Quickstart",
        "Honest Limitations & Edge Cases",
        "Project Structure & Documentation Index",
        "Testing Pyramid & Quality Gates",
    ]

    for section in expected_sections:
        assert section in content, f"Missing required section in README.md: '{section}'"


def test_readme_contains_adr_summaries() -> None:
    """Verify that all four ADRs are explicitly cited and summarized in README.md."""
    content = Path("README.md").read_text(encoding="utf-8")
    for adr in ["ADR-001", "ADR-002", "ADR-003", "ADR-004"]:
        assert adr in content, f"Missing {adr} summary in README.md"


def test_readme_contains_quantitative_ragas_metrics() -> None:
    """Verify that Ragas benchmark metrics and thresholds are present in README.md."""
    content = Path("README.md").read_text(encoding="utf-8")
    for metric in ["Context Recall", "Context Precision", "Faithfulness", "Answer Relevancy"]:
        assert metric in content, f"Missing metric '{metric}' in README.md evaluation table."


def test_readme_local_markdown_links_resolve_to_existing_files() -> None:
    """Extract all local file markdown links and assert target files exist on disk."""
    readme_path = Path("README.md")
    content = readme_path.read_text(encoding="utf-8")

    # Match markdown links: [text](link)
    # Ignore anchors (#...) and external http(s) links
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    matches = link_pattern.findall(content)

    tested_links = 0
    for text, link in matches:
        # Strip anchor fragment
        clean_link = link.split("#")[0].strip()

        # Skip web urls, mailto, or pure anchor jumps
        if not clean_link or clean_link.startswith(("http://", "https://", "mailto:")):
            continue

        target_path = Path(clean_link)
        err_msg = f"Broken link in README.md: [{text}]({link}) -> '{clean_link}' not found."
        assert target_path.exists(), err_msg
        tested_links += 1

    msg = f"Expected at least 10 local documentation links in README.md, found {tested_links}."
    assert tested_links >= 10, msg
