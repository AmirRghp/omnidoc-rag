# Task 10.1: Golden Evaluation Dataset Builder

Assemble the 30-sample curated bilingual (`fa` / `en`) technical evaluation dataset in `evals/datasets/golden_bilingual_eval.json` and implement the strict Pydantic V2 dataset loader and schema validator in `evals/eval_runner.py`, backed by unit tests in `tests/unit/test_golden_dataset.py`.

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the dataset structure and validation contract below before proceeding to implementation:

- **Strict Pydantic V2 Dataset Schema (`GoldenEvalSample`)**: Every record in `evals/datasets/golden_bilingual_eval.json` is validated with `ConfigDict(strict=True)` and includes `id`, `query`, `language` (`"fa" | "en"`), `domain` (`"docker" | "postgres" | "linux" | "out_of_domain"`), `category` (`"standard" | "flag_in_persian" | "error_log" | "ambiguous" | "out_of_domain"`), `contexts` (`list[str]`), `ground_truth` (`str`), `reference_source` (`str`), and `expected_status` (`"correct" | "ambiguous" | "incorrect"`).
- **Balanced Bilingual & Edge-Case Distribution (30 Samples)**:
  - **15 Persian (`fa`) + 15 English (`en`)** queries grounded in the existing `data/raw/*.md` corpus (`Docker`, `PostgreSQL`, `Linux`).
  - Explicitly includes the required edge cases: English CLI flags embedded in Persian queries (e.g., `--ipc=host`, `network_mode`, `shared_buffers`), raw technical error logs (e.g., `502 Bad Gateway`, Docker daemon socket permission errors, PostgreSQL lock contention), ambiguous troubleshooting queries, and out-of-domain refusal queries.
- **Forward Compatibility with Task 10.2**: `evals/eval_runner.py` exposes `load_golden_dataset()`, `validate_golden_dataset()`, and a `--validate-only` CLI mode so Task 10.2 can directly consume the validated `GoldenEvalSample` objects for Ragas metric computation without refactoring.

## 1. Overview & Core Concept

- **What It Does**: Provides the canonical 30-sample bilingual ground-truth benchmark dataset and schema verification tooling for evaluating retrieval recall/precision, grounded generation faithfulness, and CRAG refusal accuracy.
- **Target Audience / Persona**: AI/MLOps engineers and CI quality gates (`Task 10.2` & `Task 10.3`) running automated regression checks.
- **Key Value**: Prevents silent retrieval or hallucination regressions across both Persian and English technical queries.

## 2. Dataset Composition & Validation Flow

- **Dataset Breakdown (`30` total samples)**:
  - **Persian (`15` samples)**:
    - Docker (`5` samples), PostgreSQL (`4` samples), Linux (`4` samples), Out-of-Domain Refusal (`2` samples).
    - Includes mixed RTL/LTR queries with English flags (`--ipc=host`, `--network`, `max_connections`, `wal_level`, `cgroups`) and ambiguous error codes (`ارور ۵۰۲`).
  - **English (`15` samples)**:
    - Docker (`5` samples), PostgreSQL (`4` samples), Linux (`4` samples), Out-of-Domain Refusal (`2` samples).
    - Includes exact daemon/socket error logs, multi-arch Buildx, PostgreSQL MVCC/vacuum bloat, systemd container units, and out-of-domain refusal cases.

## 3. Technical Architecture & Data Strategy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                 evals/datasets/golden_bilingual_eval.json                │
│  30 Curated Samples (15 Persian 'fa' + 15 English 'en')                  │
│  Domains: Docker, PostgreSQL, Linux, Out-of-Domain                       │
│  Edge Cases: Flags in Persian, Error Logs, Ambiguous, Refusal            │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           evals/eval_runner.py                           │
├──────────────────────────────────────────────────────────────────────────┤
│  ├── GoldenEvalSample(BaseModel, strict=True)                            │
│  ├── GoldenDatasetSummary(BaseModel, strict=True)                        │
│  ├── load_golden_dataset(path: Path) -> list[GoldenEvalSample]           │
│  ├── validate_golden_dataset(samples) -> GoldenDatasetSummary            │
│  └── main(argv) -> int  (CLI verification entrypoint)                    │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     tests/unit/test_golden_dataset.py                    │
│  Verifies schema validity, 15 fa / 15 en split, domain & edge-case       │
│  coverage, unique IDs, and rejection of malformed records.               │
└──────────────────────────────────────────────────────────────────────────┘
```

### Target Files
- `evals/datasets/golden_bilingual_eval.json`
- `evals/eval_runner.py`
- `tests/unit/test_golden_dataset.py`
- `docs/Plans.md` (mark Task 10.1 `[x]`)

### Acceptance Criteria & Verification
1. **Golden Dataset (`evals/datasets/golden_bilingual_eval.json`)**:
   - Contains exactly 30 curated samples: 15 Persian (`fa`) and 15 English (`en`).
   - Covers `docker`, `postgres`, and `linux` technical documentation plus `out_of_domain` refusal cases.
   - Includes English command flags in Persian queries, technical error logs, and ambiguous queries.
2. **Dataset Validator & Runner (`evals/eval_runner.py`)**:
   - Validates all records against `GoldenEvalSample` (`ConfigDict(strict=True)`) and verifies dataset invariants (30 samples, 15 `fa` / 15 `en`, all required domains and edge-case categories present).
   - Executable via `.venv/bin/python -m evals.eval_runner` returning exit code `0`.
3. **Verification Commands**:
   - `.venv/bin/python -m evals.eval_runner` exits with `0` and prints the dataset validation summary.
   - `.venv/bin/pytest tests/unit/test_golden_dataset.py -v` passes with 0 failures.
   - `.venv/bin/ruff check .` and `.venv/bin/mypy --strict evals src app.py` pass with 0 errors.
