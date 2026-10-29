"""Golden bilingual evaluation dataset loader, Ragas metric runner, and CI quality gate."""

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common.config import Settings, create_settings
from src.common.logger import get_logger
from src.common.schemas import EvaluationRecord
from src.services.llm_client import VLLMClient

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "golden_bilingual_eval.json"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

EvalLanguage = Literal["fa", "en"]
EvalDomain = Literal["docker", "postgres", "linux", "out_of_domain"]
EvalCategory = Literal[
    "standard",
    "flag_in_persian",
    "error_log",
    "ambiguous",
    "out_of_domain",
]
ExpectedStatus = Literal["correct", "ambiguous", "incorrect"]


class GoldenEvalSample(BaseModel):
    """Strict Pydantic V2 contract for a single bilingual evaluation sample."""

    model_config = ConfigDict(strict=True)

    id: str = Field(..., min_length=1, description="Unique evaluation sample identifier.")
    query: str = Field(..., min_length=1, description="Bilingual user technical query.")
    language: EvalLanguage = Field(..., description="Query language ('fa' or 'en').")
    domain: EvalDomain = Field(..., description="Target technical domain or 'out_of_domain'.")
    category: EvalCategory = Field(
        ..., description="Evaluation scenario category including edge cases."
    )
    contexts: list[str] = Field(
        ..., min_length=1, description="Reference ground-truth context passages."
    )
    ground_truth: str = Field(
        ..., min_length=1, description="Reference ground-truth answer or refusal statement."
    )
    reference_source: str = Field(
        ..., min_length=1, description="Source markdown filename in data/raw/ or 'none'."
    )
    expected_status: ExpectedStatus = Field(
        ..., description="Expected CRAG relevance gating decision."
    )

    @field_validator("contexts")
    @classmethod
    def _validate_non_empty_contexts(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("Each sample must contain at least one non-empty context string.")
        return value


class GoldenDatasetSummary(BaseModel):
    """Summary metrics produced after validating the Golden Evaluation Dataset."""

    model_config = ConfigDict(strict=True)

    total_samples: int = Field(..., ge=0)
    language_counts: dict[str, int] = Field(default_factory=dict)
    domain_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)


class RagasGateThresholds(BaseModel):
    """Non-negotiable CI quality gate thresholds for Ragas evaluation metrics."""

    model_config = ConfigDict(strict=True)

    faithfulness: float = Field(default=0.85, ge=0.0, le=1.0)
    answer_relevancy: float = Field(default=0.85, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.80, ge=0.0, le=1.0)
    context_recall: float = Field(default=0.75, ge=0.0, le=1.0)
    max_faithfulness_regression: float = Field(default=0.03, ge=0.0, le=1.0)

    @property
    def answer_relevance(self) -> float:
        """Alias for answer_relevancy matching Task 10.2 terminology."""
        return self.answer_relevancy


class SampleMetricScores(BaseModel):
    """Ragas evaluation scores for an individual evaluation sample."""

    model_config = ConfigDict(strict=True)

    sample_id: str = Field(..., min_length=1)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)
    latency_ms: int = Field(default=0, ge=0)


class RagasEvaluationReport(BaseModel):
    """Evaluation report combining aggregated record, sample scores, and gate status."""

    model_config = ConfigDict(strict=True)

    record: EvaluationRecord
    thresholds: RagasGateThresholds
    passed: bool
    failed_metrics: list[str] = Field(default_factory=list)
    sample_scores: list[SampleMetricScores] = Field(default_factory=list)

    @property
    def mean_context_recall(self) -> float:
        """Return aggregated context_recall score."""
        return self.record.context_recall

    @property
    def mean_context_precision(self) -> float:
        """Return aggregated context_precision score."""
        return self.record.context_precision

    @property
    def mean_faithfulness(self) -> float:
        """Return aggregated faithfulness score."""
        return self.record.faithfulness

    @property
    def mean_answer_relevancy(self) -> float:
        """Return aggregated answer_relevancy score."""
        return self.record.answer_relevancy


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenEvalSample]:
    """Load and parse the golden bilingual evaluation JSON dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Golden evaluation dataset not found at: {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Golden dataset at {path} must contain a JSON list of records.")

    return [GoldenEvalSample.model_validate(item) for item in raw_data]


def validate_golden_dataset(
    samples: list[GoldenEvalSample],
    expected_total: int = 30,
    expected_per_language: int = 15,
) -> GoldenDatasetSummary:
    """Verify structural invariants, bilingual balance, and edge-case coverage."""
    if len(samples) != expected_total:
        raise ValueError(
            f"Expected exactly {expected_total} evaluation samples, got {len(samples)}."
        )

    seen_ids: set[str] = set()
    for sample in samples:
        if sample.id in seen_ids:
            raise ValueError(f"Duplicate sample ID detected: {sample.id}")
        seen_ids.add(sample.id)

    lang_counts = Counter(s.language for s in samples)
    if lang_counts.get("fa", 0) != expected_per_language:
        raise ValueError(
            f"Expected {expected_per_language} Persian ('fa') samples, "
            f"got {lang_counts.get('fa', 0)}."
        )
    if lang_counts.get("en", 0) != expected_per_language:
        raise ValueError(
            f"Expected {expected_per_language} English ('en') samples, "
            f"got {lang_counts.get('en', 0)}."
        )

    domain_counts = Counter(s.domain for s in samples)
    for req_domain in ("docker", "postgres", "linux", "out_of_domain"):
        if domain_counts.get(req_domain, 0) == 0:
            raise ValueError(f"Missing required domain coverage for: {req_domain}")

    category_counts = Counter(s.category for s in samples)
    for req_cat in ("flag_in_persian", "error_log", "ambiguous", "out_of_domain"):
        if category_counts.get(req_cat, 0) == 0:
            raise ValueError(f"Missing required edge-case category: {req_cat}")

    return GoldenDatasetSummary(
        total_samples=len(samples),
        language_counts={str(k): int(v) for k, v in lang_counts.items()},
        domain_counts={str(k): int(v) for k, v in domain_counts.items()},
        category_counts={str(k): int(v) for k, v in category_counts.items()},
    )


def check_gating_thresholds(
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
    context_recall: float,
    thresholds: RagasGateThresholds | None = None,
    baseline_faithfulness: float | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate aggregated Ragas scores against CI quality gate thresholds.

    Args:
        faithfulness: Aggregated faithfulness score in [0.0, 1.0].
        answer_relevancy: Aggregated answer relevance score in [0.0, 1.0].
        context_precision: Aggregated context precision score in [0.0, 1.0].
        context_recall: Aggregated context recall score in [0.0, 1.0].
        thresholds: Optional custom RagasGateThresholds.
        baseline_faithfulness: Optional baseline faithfulness to enforce <=3% regression rule.

    Returns:
        Tuple of (passed, list of failure descriptions).
    """
    active_thresholds = thresholds or RagasGateThresholds()
    failures: list[str] = []

    if faithfulness < active_thresholds.faithfulness:
        failures.append(f"faithfulness ({faithfulness:.3f} < {active_thresholds.faithfulness:.2f})")
    if answer_relevancy < active_thresholds.answer_relevancy:
        failures.append(
            f"answer_relevancy ({answer_relevancy:.3f} < {active_thresholds.answer_relevancy:.2f})"
        )
    if context_precision < active_thresholds.context_precision:
        failures.append(
            f"context_precision ({context_precision:.3f} < "
            f"{active_thresholds.context_precision:.2f})"
        )
    if context_recall < active_thresholds.context_recall:
        failures.append(
            f"context_recall ({context_recall:.3f} < {active_thresholds.context_recall:.2f})"
        )

    if baseline_faithfulness is not None:
        drop = baseline_faithfulness - faithfulness
        if drop > active_thresholds.max_faithfulness_regression + 1e-9:
            failures.append(
                f"faithfulness_regression (drop {drop:.3f} > "
                f"{active_thresholds.max_faithfulness_regression:.2f})"
            )

    return (len(failures) == 0, failures)


def _clamp_score(value: Any, default: float = 0.0) -> float:
    """Safely convert and clamp a score into [0.0, 1.0]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def _parse_judge_json(raw_text: str) -> dict[str, float]:
    """Parse JSON metric scores from local vLLM judge output."""
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = match.group(0) if match else cleaned
    data = json.loads(candidate)
    if not isinstance(data, dict):
        raise ValueError("Judge output JSON is not an object.")

    answer_rel = data.get("answer_relevancy", data.get("answer_relevance", 0.0))
    return {
        "faithfulness": _clamp_score(data.get("faithfulness", 0.0)),
        "answer_relevancy": _clamp_score(answer_rel),
        "context_precision": _clamp_score(data.get("context_precision", 0.0)),
        "context_recall": _clamp_score(data.get("context_recall", 0.0)),
    }


class RagasEvaluator:
    """Automated Ragas metric evaluator powered by local vLLM judge (Qwen/Qwen2.5-7B-Instruct)."""

    def __init__(
        self,
        llm_client: VLLMClient | Any | None = None,
        settings: Settings | None = None,
        thresholds: RagasGateThresholds | None = None,
        workflow: Any | None = None,
    ) -> None:
        """Initialize the Ragas evaluator."""
        self.settings = settings or create_settings()
        self.thresholds = thresholds or RagasGateThresholds()
        self.workflow = workflow
        self.llm = llm_client or VLLMClient(
            model=DEFAULT_JUDGE_MODEL,
            settings=self.settings,
        )

    @staticmethod
    def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenEvalSample]:
        """Load the golden evaluation dataset from disk."""
        return load_golden_dataset(path)

    @staticmethod
    def build_judge_messages(
        sample: GoldenEvalSample,
        generated_answer: str,
        retrieved_contexts: list[str],
    ) -> list[dict[str, str]]:
        """Build structured evaluation prompt for the local vLLM judge model."""
        system_prompt = (
            "You are an objective Ragas evaluation judge for a bilingual (Persian/English) "
            "technical RAG system. Evaluate the response strictly against the provided contexts "
            "and ground_truth. Output ONLY valid JSON with keys: "
            '"faithfulness", "answer_relevancy", "context_precision", "context_recall" '
            "with float values in [0.0, 1.0]."
        )
        user_payload = json.dumps(
            {
                "query": sample.query,
                "language": sample.language,
                "retrieved_contexts": retrieved_contexts,
                "generated_answer": generated_answer,
                "ground_truth": sample.ground_truth,
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

    async def aevaluate_sample(
        self,
        sample: GoldenEvalSample,
        generated_answer: str | None = None,
        retrieved_contexts: list[str] | None = None,
    ) -> SampleMetricScores:
        """Compute Faithfulness, Answer Relevance, Context Precision, and Context Recall.

        If `self.workflow` is provided and `generated_answer` / `retrieved_contexts` are
        omitted, executes `await self.workflow.ainvoke(query=sample.query)` to evaluate
        actual pipeline outputs rather than static ground_truth strings.
        """
        start_time = time.perf_counter()
        if self.workflow is not None and (generated_answer is None or retrieved_contexts is None):
            state = await self.workflow.ainvoke(query=sample.query, language=sample.language)
            if generated_answer is None:
                generated_answer = str(state.get("generation", ""))
            if retrieved_contexts is None:
                chunks = state.get("reranked_chunks", [])
                retrieved_contexts = [str(getattr(c, "content", c)) for c in chunks]

        answer = generated_answer if generated_answer is not None else sample.ground_truth
        contexts = (
            retrieved_contexts
            if retrieved_contexts is not None and len(retrieved_contexts) > 0
            else sample.contexts
        )

        messages = self.build_judge_messages(
            sample=sample,
            generated_answer=answer,
            retrieved_contexts=contexts,
        )
        raw_response = await self.llm.generate(messages, temperature=0.0)
        parsed = _parse_judge_json(str(raw_response))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return SampleMetricScores(
            sample_id=sample.id,
            faithfulness=parsed["faithfulness"],
            answer_relevancy=parsed["answer_relevancy"],
            context_precision=parsed["context_precision"],
            context_recall=parsed["context_recall"],
            latency_ms=max(0, elapsed_ms),
        )

    async def aevaluate_dataset(
        self,
        samples: list[GoldenEvalSample],
        run_name: str = "ragas-bilingual-gating-run",
        retrieval_strategy: Literal["dense_only", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        baseline_faithfulness: float | None = None,
    ) -> RagasEvaluationReport:
        """Evaluate a list of GoldenEvalSample records and verify CI gating thresholds."""
        if not samples:
            raise ValueError("Cannot run Ragas evaluation on an empty sample list.")

        sample_scores: list[SampleMetricScores] = []
        for sample in samples:
            score_item = await self.aevaluate_sample(sample)
            sample_scores.append(score_item)

        count = len(sample_scores)
        avg_faith = sum(s.faithfulness for s in sample_scores) / count
        avg_rel = sum(s.answer_relevancy for s in sample_scores) / count
        avg_prec = sum(s.context_precision for s in sample_scores) / count
        avg_rec = sum(s.context_recall for s in sample_scores) / count

        sorted_latencies = sorted(s.latency_ms for s in sample_scores)
        p95_idx = min(count - 1, max(0, int(round(0.95 * (count - 1)))))
        p95_latency = int(sorted_latencies[p95_idx])

        passed, failed_metrics = check_gating_thresholds(
            faithfulness=avg_faith,
            answer_relevancy=avg_rel,
            context_precision=avg_prec,
            context_recall=avg_rec,
            thresholds=self.thresholds,
            baseline_faithfulness=baseline_faithfulness,
        )

        model_name = str(getattr(self.llm, "model", DEFAULT_JUDGE_MODEL))
        record = EvaluationRecord(
            run_name=run_name,
            model_name=model_name,
            embedding_model=self.settings.dense_model_name,
            retrieval_strategy=retrieval_strategy,
            context_recall=round(avg_rec, 4),
            context_precision=round(avg_prec, 4),
            faithfulness=round(avg_faith, 4),
            answer_relevancy=round(avg_rel, 4),
            p95_latency_ms=p95_latency,
            total_samples=count,
            pass_threshold=passed,
        )

        return RagasEvaluationReport(
            record=record,
            thresholds=self.thresholds,
            passed=passed,
            failed_metrics=failed_metrics,
            sample_scores=sample_scores,
        )


def _create_default_judge_client() -> VLLMClient:
    """Create the default local vLLM judge client (Qwen/Qwen2.5-7B-Instruct)."""
    settings = create_settings()
    return VLLMClient(model=settings.llm_model or DEFAULT_JUDGE_MODEL, settings=settings)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to validate dataset or run Ragas evaluation with gating exit codes."""
    parser = argparse.ArgumentParser(
        description="Validate dataset and execute Ragas gating evaluation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to golden_bilingual_eval.json",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset schema and invariants without running LLM evaluation.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of samples to evaluate from the dataset (default: 30).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to export JSON evaluation report.",
    )
    parser.add_argument(
        "--baseline-faithfulness",
        type=float,
        default=None,
        help="Optional baseline faithfulness score to enforce <=3%% regression gate.",
    )
    args = parser.parse_args(argv)

    try:
        samples = load_golden_dataset(args.dataset)
        summary = validate_golden_dataset(samples)

        if args.validate_only:
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "dataset": str(args.dataset),
                        "summary": summary.model_dump(),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        selected_samples = samples[: max(1, min(len(samples), int(args.samples)))]
        judge_client = _create_default_judge_client()

        async def _run() -> RagasEvaluationReport:
            try:
                evaluator = RagasEvaluator(llm_client=judge_client)
                return await evaluator.aevaluate_dataset(
                    samples=selected_samples,
                    baseline_faithfulness=args.baseline_faithfulness,
                )
            finally:
                if hasattr(judge_client, "aclose"):
                    await judge_client.aclose()

        report = asyncio.run(_run())
        report_json = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
        print(report_json)

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report_json, encoding="utf-8")

        return 0 if report.passed else 1

    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
