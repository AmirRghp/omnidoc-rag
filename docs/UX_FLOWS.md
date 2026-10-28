# User Experience & System Flows (UX_FLOWS.md)

## Overview
This document outlines the end-to-end user journeys through OmniDoc RAG, capturing both the human-facing UI transitions and the corresponding background state-machine events.

---

## Flow 1 — Standard Technical Q&A (Happy Path)
*User asks a valid technical question found within the indexed technical documentation.*

```
[User enters prompt] 
       │
       ▼
[UI displays "Searching documents..."]
       ├── Dense vector search (Qdrant)
       └── Lexical sparse search (BM25)
       │
       ▼
[UI displays "Fusing & Re-ranking 15 candidates..."]
       ├── RRF reciprocal rank fusion
       └── Cross-Encoder rescores top candidates
       │
       ▼
[UI displays "Verifying context relevance (CRAG)..."]
       └── Evaluator scores context >= 0.70 (Status: CORRECT)
       │
       ▼
[Token Streaming Begins]
       ├── Real-time SSE token stream
       └── Interactive Footnote Citations rendered ([Source: docker_compose.md#L45])
       │
       ▼
[Complete] User can expand "Reasoning Steps" accordion or copy code snippets.
```

- **Inputs:** Natural language query (e.g., "نحوه تنظیم network_mode در داکر کامپوز چیست؟").
- **Loading State:** Stepper accordion opens automatically, pulsing current execution phase with millisecond timers.
- **Success State:** Markdown formatted response with syntax-highlighted code blocks and interactive pills linking directly to referenced document chunks.

---

## Flow 2 — Corrective RAG (Query Ambiguity & Automatic Rewrite)
*User submits a brief, vague, or slang query that initially yields weak retrieval scores.*

```
[User Query: "مشکل ارور ۵۰۲"]
       │
       ▼
[Initial Hybrid Retrieval: Low relevance scores (Max Cross-Encoder Score: 0.38)]
       │
       ▼
[CRAG Evaluator Node Triggered: Decision = AMBIGUOUS]
       │
       ▼
[UI displays Badge: "Refining search query for better technical context..."]
       │
       ▼
[LLM Query Rewriter]:
       └── Formulates: "Docker reverse proxy Nginx HTTP 502 Bad Gateway upstream connection failure"
       │
       ▼
[Secondary Retrieval with Rewritten Query]
       └── Top candidates retrieved with Score: 0.82
       │
       ▼
[Generation & Streaming with Source Attribution]
```

- **Edge Case / User Feedback:** UI shows an alert: *"کوئری شما برای دقت بیشتر به عبارت تخصصی‌تر بازنویسی شد."* (Your query was rewritten for higher precision).
- **Failure Boundary:** Maximum 1 rewrite loop to avoid infinite latency or execution loops.

---

## Flow 3 — Out-of-Domain / Insufficient Documentation (Hallucination Defense)
*User asks a question not covered in the local technical documentation.*

```
[User Query: "قیمت روز سهام اپل چقدر است؟"]
       │
       ▼
[Retrieval & Cross-Encoder: All scores < 0.25]
       │
       ▼
[CRAG Evaluator Node Triggered: Decision = INCORRECT / EMPTY]
       │
       ▼
[Controlled Refusal Generation (Zero Hallucination)]
```

- **Output Message:** *"پاسخ این سوال در مستندات بارگذاری‌شده موجود نیست. سیستم تنها بر مبنای مستندات فنی رسمی پاسخ می‌دهد و از حدس‌زدن خودداری می‌کند."*
- **UI State:** Neutral warning alert, no hallucinated steps, option to upload additional documentation.

---

## Flow 4 — Document Ingestion & Re-indexing Flow
*Admin/Developer ingests new markdown documentation.*

```
[Upload / Trigger Ingest: CLI or REST API]
       │
       ▼
[Validation: Format check (.md/.pdf), encoding check (UTF-8), size <= 25MB]
       │
       ▼
[Semantic Chunker: Section/Header-aware splitting + Metadata extraction]
       │
       ▼
[Dual Indexing Pipeline]:
       ├── Dense Embedding Generation (BGE-M3) ──► Batched Upsert to Qdrant
       └── Tokenization & Inverted Index ─────────► BM25 Index Persistence
       │
       ▼
[Redis Cache Invalidation for stale documents]
       │
       ▼
[Ingestion Summary Report: Chunks count, Token volume, Indexing duration]
```

---

## Flow 5 — Automated Benchmark & Evaluation Run (Ragas)
*Developer runs automated regression testing before pushing code.*

```
[Developer triggers: `python -m evals.run_benchmark`]
       │
       ▼
[Load Golden Dataset (50 QA items from data/golden_eval_dataset.json)]
       │
       ▼
[Execute pipeline in batch with vLLM local serving]
       │
       ▼
[Ragas Engine calculates: Context Recall, Precision, Faithfulness, Relevancy]
       │
       ▼
[Comparison Table generated against baseline]
       ├── If Faithfulness < 0.85 ──► Exit Code 1 (CI Failure)
       └── If Metrics meet target ──► Export report to evals/results/latest.json & Langfuse
```
