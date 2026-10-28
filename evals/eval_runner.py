"""Golden bilingual evaluation dataset loader, schema validator, and CLI runner (Task 10.1)."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "golden_bilingual_eval.json"

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


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenEvalSample]:
    """Load and parse the golden bilingual evaluation JSON dataset.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        Validated list of GoldenEvalSample records.

    Raises:
        FileNotFoundError: If dataset file does not exist.
        ValueError: If JSON root is not a list.
    """
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
    """Verify structural invariants, bilingual balance, and edge-case coverage.

    Args:
        samples: Parsed GoldenEvalSample list.
        expected_total: Expected total sample count (default: 30).
        expected_per_language: Expected count per language 'fa' and 'en' (default: 15).

    Returns:
        GoldenDatasetSummary with distribution counts.

    Raises:
        ValueError: If any dataset invariant is violated.
    """
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint to validate the Golden Bilingual Evaluation Dataset."""
    parser = argparse.ArgumentParser(
        description="Validate and inspect the OmniDoc RAG Golden Bilingual Evaluation Dataset."
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
    args = parser.parse_args(argv)

    try:
        samples = load_golden_dataset(args.dataset)
        summary = validate_golden_dataset(samples)
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
    except Exception as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
