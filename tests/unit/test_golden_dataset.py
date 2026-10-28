"""Unit tests for the Golden Bilingual Evaluation Dataset and schema validator (Task 10.1)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.eval_runner import (
    DEFAULT_DATASET_PATH,
    GoldenEvalSample,
    load_golden_dataset,
    main,
    validate_golden_dataset,
)


def test_golden_dataset_file_exists_and_loads() -> None:
    """Verify evals/datasets/golden_bilingual_eval.json exists and loads 30 valid samples."""
    assert DEFAULT_DATASET_PATH.is_file(), f"Missing dataset at {DEFAULT_DATASET_PATH}"
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)
    assert len(samples) == 30
    assert all(isinstance(s, GoldenEvalSample) for s in samples)


def test_golden_dataset_bilingual_balance_and_domains() -> None:
    """Verify 15 Persian ('fa') and 15 English ('en') samples across required domains."""
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)
    summary = validate_golden_dataset(samples)

    assert summary.total_samples == 30
    assert summary.language_counts["fa"] == 15
    assert summary.language_counts["en"] == 15

    # Core technical domains and out-of-domain refusal coverage
    assert summary.domain_counts.get("docker", 0) >= 6
    assert summary.domain_counts.get("postgres", 0) >= 6
    assert summary.domain_counts.get("linux", 0) >= 6
    assert summary.domain_counts.get("out_of_domain", 0) >= 2


def test_golden_dataset_includes_required_edge_cases() -> None:
    """Verify tricky edge cases: English flags in Persian, error logs, ambiguous, refusal."""
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)
    summary = validate_golden_dataset(samples)

    assert summary.category_counts.get("flag_in_persian", 0) >= 3
    assert summary.category_counts.get("error_log", 0) >= 3
    assert summary.category_counts.get("ambiguous", 0) >= 2
    assert summary.category_counts.get("out_of_domain", 0) >= 2

    # Check that flag_in_persian samples are actually in Persian and contain ASCII flags/identifiers
    flag_samples = [s for s in samples if s.category == "flag_in_persian"]
    for sample in flag_samples:
        assert sample.language == "fa"
        assert any(
            token in sample.query
            for token in ("--", "_", "network_mode", "shared_buffers", "wal_level", "cgroups")
        )

    # Check that out_of_domain samples expect 'incorrect' CRAG status
    ood_samples = [s for s in samples if s.category == "out_of_domain"]
    for sample in ood_samples:
        assert sample.expected_status == "incorrect"
        assert sample.domain == "out_of_domain"


def test_golden_eval_sample_strict_schema_rejects_invalid_records(tmp_path: Path) -> None:
    """Verify strict Pydantic V2 validation and dataset invariant checks on malformed inputs."""
    with pytest.raises(ValidationError):
        GoldenEvalSample(
            id="bad-1",
            query="",  # empty query must fail min_length=1
            language="fa",
            domain="docker",
            category="standard",
            contexts=["some context"],
            ground_truth="some truth",
            reference_source="docker_overview.md",
            expected_status="correct",
        )

    # Unbalanced dataset should fail validate_golden_dataset
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)
    with pytest.raises(ValueError, match="Expected exactly 30"):
        validate_golden_dataset(samples[:10])

    # Duplicate IDs should fail validate_golden_dataset
    duplicated = [samples[0], samples[0], *samples[2:]]
    with pytest.raises(ValueError, match="Duplicate sample ID"):
        validate_golden_dataset(duplicated)

    # Non-list JSON payload should fail load_golden_dataset
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON list"):
        load_golden_dataset(bad_json)


def test_eval_runner_cli_validate_only() -> None:
    """Verify CLI main entrypoint returns exit code 0 on valid golden dataset."""
    exit_code = main(["--dataset", str(DEFAULT_DATASET_PATH), "--validate-only"])
    assert exit_code == 0
