"""Integration and gating tests for Ragas automated metric evaluation (Task 10.2)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from evals.eval_runner import (
    DEFAULT_DATASET_PATH,
    RagasEvaluationReport,
    RagasEvaluator,
    RagasGateThresholds,
    check_gating_thresholds,
    load_golden_dataset,
    main,
)


@pytest.mark.eval
def test_default_ragas_gating_thresholds_match_spec() -> None:
    """Verify default gating thresholds match Task 10.2 specification."""
    thresholds = RagasGateThresholds()
    assert thresholds.faithfulness == 0.85
    assert thresholds.answer_relevancy == 0.85
    assert thresholds.answer_relevance == 0.85
    assert thresholds.context_precision == 0.80
    assert thresholds.context_recall == 0.75

    passed, failed = check_gating_thresholds(
        faithfulness=0.92,
        answer_relevancy=0.89,
        context_precision=0.86,
        context_recall=0.88,
        thresholds=thresholds,
    )
    assert passed is True
    assert failed == []


@pytest.mark.eval
@pytest.mark.parametrize(
    ("faith", "rel", "prec", "rec", "expected_failed"),
    [
        (0.84, 0.90, 0.85, 0.80, "faithfulness"),
        (0.90, 0.82, 0.85, 0.80, "answer_relevancy"),
        (0.90, 0.90, 0.78, 0.80, "context_precision"),
        (0.90, 0.90, 0.85, 0.72, "context_recall"),
    ],
)
def test_check_gating_thresholds_detects_each_failing_metric(
    faith: float,
    rel: float,
    prec: float,
    rec: float,
    expected_failed: str,
) -> None:
    """Verify check_gating_thresholds fails when any metric drops below its floor."""
    passed, failed = check_gating_thresholds(
        faithfulness=faith,
        answer_relevancy=rel,
        context_precision=prec,
        context_recall=rec,
    )
    assert passed is False
    assert any(expected_failed in item for item in failed)


@pytest.mark.eval
def test_check_gating_thresholds_baseline_faithfulness_regression() -> None:
    """Verify >3% faithfulness drop relative to baseline triggers gating failure."""
    passed, failed = check_gating_thresholds(
        faithfulness=0.89,
        answer_relevancy=0.90,
        context_precision=0.85,
        context_recall=0.82,
        baseline_faithfulness=0.94,
    )
    assert passed is False
    assert any("faithfulness_regression" in item for item in failed)


@pytest.mark.eval
@pytest.mark.asyncio
async def test_ragas_evaluator_with_local_vllm_judge_passes_gate() -> None:
    """Verify RagasEvaluator scores bilingual samples via Qwen/Qwen2.5-7B-Instruct judge."""
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)[:4]

    mock_llm = MagicMock()
    mock_llm.model = "Qwen/Qwen2.5-7B-Instruct"
    mock_llm.generate = AsyncMock(
        return_value=json.dumps(
            {
                "faithfulness": 0.93,
                "answer_relevancy": 0.91,
                "context_precision": 0.88,
                "context_recall": 0.86,
            }
        )
    )

    evaluator = RagasEvaluator(llm_client=mock_llm)
    report: RagasEvaluationReport = await evaluator.aevaluate_dataset(
        samples=samples,
        run_name="test-bilingual-gate-pass",
    )

    assert report.passed is True
    assert report.failed_metrics == []
    assert report.record.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert report.record.total_samples == 4
    assert report.record.pass_threshold is True
    assert report.record.faithfulness == pytest.approx(0.93)
    assert report.record.answer_relevancy == pytest.approx(0.91)
    assert report.record.context_precision == pytest.approx(0.88)
    assert report.record.context_recall == pytest.approx(0.86)


@pytest.mark.eval
@pytest.mark.asyncio
async def test_ragas_evaluator_fails_gate_when_judge_scores_below_threshold() -> None:
    """Verify RagasEvaluator marks report as failed when judge scores drop below threshold."""
    samples = load_golden_dataset(DEFAULT_DATASET_PATH)[:2]

    mock_llm = MagicMock()
    mock_llm.model = "Qwen/Qwen2.5-7B-Instruct"
    mock_llm.generate = AsyncMock(
        return_value=json.dumps(
            {
                "faithfulness": 0.72,
                "answer_relevancy": 0.88,
                "context_precision": 0.84,
                "context_recall": 0.79,
            }
        )
    )

    evaluator = RagasEvaluator(llm_client=mock_llm)
    report = await evaluator.aevaluate_dataset(
        samples=samples,
        run_name="test-bilingual-gate-fail",
    )

    assert report.passed is False
    assert report.record.pass_threshold is False
    assert any("faithfulness" in m for m in report.failed_metrics)


@pytest.mark.eval
def test_eval_runner_cli_exit_codes_for_pass_and_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI returns exit code 0 when thresholds pass and exit code 1 when any fails."""
    output_file = tmp_path / "eval_report.json"

    # Case 1: High scores -> exit code 0
    monkeypatch.setattr(
        "evals.eval_runner._create_default_judge_client",
        lambda: _build_stub_vllm_client(0.92, 0.90, 0.86, 0.82),
    )
    exit_ok = main(
        [
            "--dataset",
            str(DEFAULT_DATASET_PATH),
            "--samples",
            "4",
            "--output",
            str(output_file),
        ]
    )
    assert exit_ok == 0
    assert output_file.is_file()
    saved_report = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_report["passed"] is True

    # Case 2: Low faithfulness (0.78 < 0.85) -> exit code 1
    monkeypatch.setattr(
        "evals.eval_runner._create_default_judge_client",
        lambda: _build_stub_vllm_client(0.78, 0.90, 0.86, 0.82),
    )
    exit_fail = main(
        [
            "--dataset",
            str(DEFAULT_DATASET_PATH),
            "--samples",
            "4",
            "--output",
            str(output_file),
        ]
    )
    assert exit_fail == 1


def _build_stub_vllm_client(
    faith: float,
    rel: float,
    prec: float,
    rec: float,
) -> MagicMock:
    """Create a stub VLLMClient returning deterministic JSON metric scores."""
    stub = MagicMock()
    stub.model = "Qwen/Qwen2.5-7B-Instruct"
    stub.generate = AsyncMock(
        return_value=json.dumps(
            {
                "faithfulness": faith,
                "answer_relevancy": rel,
                "context_precision": prec,
                "context_recall": rec,
            }
        )
    )
    stub.aclose = AsyncMock()
    return stub
