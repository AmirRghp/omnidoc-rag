# Task 10.2: Ragas Automated Metric Evaluation Runner

Extend `evals/eval_runner.py` to execute quantitative Ragas evaluation (`Faithfulness`, `Answer Relevance`, `Context Precision`, and `Context Recall`) using the local vLLM judge model (`Qwen/Qwen2.5-7B-Instruct`), enforce CI quality gate thresholds, and add the `@pytest.mark.eval` integration suite in `tests/integration/test_ragas_gating.py`.

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the evaluation runner architecture and gating thresholds below before proceeding to implementation:

- **Enforced Quality Gate Thresholds (`RagasGateThresholds`)**:
  - `faithfulness` $\ge 0.85$
  - `answer_relevancy` (`answer_relevance`) $\ge 0.85$
  - `context_precision` $\ge 0.80$
  - `context_recall` $\ge 0.75$
  - Plus optional baseline regression guard (`docs/TESTING.md` §4.3): fails if `faithfulness` drops by $> 0.03$ ($3\%$) relative to an optional baseline score.
- **Local vLLM Judge (`Qwen/Qwen2.5-7B-Instruct`) with Deterministic Fallback & Dependency Injection**:
  - `RagasEvaluator` uses `VLLMClient` (`src/services/llm_client.py`) targeting `settings.llm_model` (`Qwen/Qwen2.5-7B-Instruct`) to score each sample's `(query, retrieved_contexts, generated_answer, ground_truth)` across the 4 Ragas metrics, and supports injecting a custom judge client or workflow for fast, deterministic testing in `tests/integration/test_ragas_gating.py`.
- **Reusing Existing Domain Contracts**:
  - Produces a validated `EvaluationRecord` from `src.common.schemas` alongside a detailed `RagasEvaluationReport` and exits with code `1` whenever any metric violates its threshold (or `0` when all gates pass).
- **Pytest `--strict-markers` Registration**:
  - Registers the `eval` marker in `pyproject.toml` under `[tool.pytest.ini_options]` so `pytest tests/integration/test_ragas_gating.py -v -m eval` runs cleanly under `--strict-markers`.

## 1. Overview & Core Concept

- **What It Does**: Evaluates RAG pipeline outputs against the 30-sample bilingual Golden Dataset (`evals/datasets/golden_bilingual_eval.json`), aggregates `Faithfulness`, `Answer Relevance`, `Context Precision`, and `Context Recall`, and enforces non-negotiable CI gating floors.
- **Target Audience / Persona**: MLOps engineers and automated CI pipelines (`Task 10.3`) verifying that prompt, chunking, or retrieval changes do not introduce hallucinations or retrieval regressions.
- **Key Value**: Converts qualitative RAG quality into an automated, deterministic pass/fail gate with exit code `0` / `1`.

## 2. Evaluation & Gating Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│           Golden Dataset (evals/datasets/golden_bilingual_eval.json)     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 RagasEvaluator (evals/eval_runner.py)                    │
│  ├── Judge Model: Local vLLM (Qwen/Qwen2.5-7B-Instruct via VLLMClient)   │
│  ├── Per-Sample Scoring:                                                 │
│  │     • faithfulness        ∈ [0.0, 1.0]                                │
│  │     • answer_relevancy    ∈ [0.0, 1.0]                                │
│  │     • context_precision   ∈ [0.0, 1.0]                                │
│  │     • context_recall      ∈ [0.0, 1.0]                                │
│  └── Aggregation -> EvaluationRecord (src.common.schemas)                │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       check_gating_thresholds(...)                       │
│  • Faithfulness      >= 0.85   • Context Precision >= 0.80               │
│  • Answer Relevance  >= 0.85   • Context Recall    >= 0.75               │
│  • Faithfulness drop <= 0.03 vs baseline (if baseline provided)          │
├────────────────────────────────────┬─────────────────────────────────────┤
│  All Pass -> Exit Code 0           │  Any Fail -> Exit Code 1            │
└────────────────────────────────────┴─────────────────────────────────────┘
```

## 3. Target Files & Signatures

- **`evals/eval_runner.py`**:
  - `RagasGateThresholds(BaseModel)` — defaults: `faithfulness=0.85`, `answer_relevancy=0.85`, `context_precision=0.80`, `context_recall=0.75`, `max_faithfulness_regression=0.03`.
  - `SampleMetricScores(BaseModel)` — per-sample scores (`sample_id`, `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `latency_ms`).
  - `RagasEvaluationReport(BaseModel)` — wraps `record: EvaluationRecord`, `thresholds: RagasGateThresholds`, `passed: bool`, `failed_metrics: list[str]`, and `sample_scores: list[SampleMetricScores]`.
  - `check_gating_thresholds(faithfulness, answer_relevancy, context_precision, context_recall, thresholds, baseline_faithfulness=None) -> tuple[bool, list[str]]`
  - `RagasEvaluator` — async `aevaluate_sample(...)` and `aevaluate_dataset(...)` using `VLLMClient` (`Qwen/Qwen2.5-7B-Instruct`).
  - `main(argv)` — supports `--validate-only`, `--samples`, `--output`, and `--baseline-faithfulness`, returning exit code `0` on pass and `1` on gating failure or error.
- **`tests/integration/test_ragas_gating.py`**:
  - `@pytest.mark.eval` tests covering:
    1. Passing evaluation run above all 4 thresholds ($\rightarrow$ `passed=True`, exit code `0`).
    2. Failing gating scenarios for each individual metric (`faithfulness < 0.85`, `answer_relevancy < 0.85`, `context_precision < 0.80`, `context_recall < 0.75`) and baseline regression $> 3\%$ ($\rightarrow$ `passed=False`, exit code `1`).
    3. End-to-end `RagasEvaluator.aevaluate_dataset` with mocked local vLLM judge (`Qwen/Qwen2.5-7B-Instruct`) across bilingual samples and JSON report export.
- **`pyproject.toml`**:
  - Register `markers = ["eval: Quantitative Ragas evaluation and gating suite"]` under `[tool.pytest.ini_options]`.
- **`docs/Plans.md`**:
  - Mark Task 10.2 as completed (`[x]`).

## 4. Acceptance Criteria & Verification

1. **Gating Thresholds Enforced**:
   - `Faithfulness >= 0.85`, `Answer Relevance >= 0.85`, `Context Precision >= 0.80`, `Context Recall >= 0.75`.
   - CLI `main()` returns exit code `1` whenever any metric falls below its threshold and `0` when all thresholds are met.
2. **Local vLLM Judge Integration**:
   - Uses `Qwen/Qwen2.5-7B-Instruct` via `VLLMClient` to parse structured JSON metric judgments with strict $[0.0, 1.0]$ bounds clamping.
3. **Verification Commands**:
   - `.venv/bin/pytest tests/integration/test_ragas_gating.py -v -m eval` passes with 0 failures.
   - `.venv/bin/ruff check .` and `.venv/bin/mypy --strict evals src app.py` pass with 0 errors.
