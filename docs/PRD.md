# Product Requirements Document (PRD)

## 1. Product Overview
- **Product Name:** OmniDoc RAG (Robust Bilingual Technical RAG & Evaluation System)
- **One-Line Description:** An enterprise-ready, bilingual (Persian/English) technical retrieval-augmented generation engine with hybrid search, corrective reflection (CRAG), and automated evaluation.
- **Product Type:** Developer Tool / Enterprise Infrastructure / Technical Assistant
- **Primary Platform:** Web Application (FastAPI REST API + Chainlit UI + Dockerized Microservices)

---

## 2. Problem
### What problem are we solving?
1. **The Retrieval Failure Gap:** Naive vector search frequently misses exact matches like error codes (e.g., `ERR_CONNECTION_REFUSED`), system flags (`--ipc=host`), and exact Persian technical terminology.
2. **Hallucination on Irrelevant Context:** When vector search retrieves low-quality passages, standard LLMs generate convincing yet incorrect answers instead of acknowledging missing information.
3. **The "Demo vs. Production" Evaluation Gap:** Most RAG tutorials lack quantitative evaluation. Teams have no automated way to know if changing a prompt or chunking strategy regresses retrieval or answer quality.
4. **Bilingual Friction:** Persian and English code snippets mixed together frequently break tokenization, sentence boundaries, and semantic ranking in standard mono-lingual pipelines.

### Why does this problem matter?
In DevOps and system engineering, a wrong command or misdiagnosed error code can cause production downtime. High retrieval precision, zero hallucination on missing context, and deterministic performance are mandatory.

### Current Alternatives & Limitations
- *Standard LangChain/LlamaIndex Tutorials:* Naive dense search only, no automated evaluation, hardcoded cloud API dependency.
- *Proprietary Enterprise Search:* High per-seat pricing, data privacy concerns with cloud APIs, poor handling of mixed Persian-English technical text.

---

## 3. Goals
### Primary Goal
Build a robust, fully local-capable, production-grade Hybrid RAG system that achieves ≥90% retrieval recall and ≥92% faithfulness on technical documentation, verified by automated Ragas benchmarks.

### Secondary Goals
- Provide zero external API cost via local vLLM serving.
- Implement an automated CI evaluation gate to catch quality regressions.
- Offer an interactive, real-time UI showing intermediate thought steps (retrieval scores, reranking, reflection).

---

## 4. Target Users
### User Type 1: DevOps & Backend Engineers
- **Description:** Engineers diagnosing infrastructure issues, reading Docker/PostgreSQL/Kubernetes manuals.
- **Needs:** Rapid lookup of error codes, configuration flags, and architectural best practices in Persian or English.
- **Pain Points:** Search engines return fragmented blog posts; standard LLMs hallucinate non-existent flags.

### User Type 2: AI / MLOps Hiring Managers & Technical Leads
- **Description:** Reviewers assessing the candidate's engineering maturity.
- **Needs:** Clear architectural decisions, reproducible benchmarks, defensive programming, and production hardening.
- **Pain Points:** Portfolios filled with trivial wrapper scripts with no evaluation or error handling.

---

## 5. User Stories
- **US-01 (Technical Query):** As an engineer, I want to paste a complex error log or query (in Persian or English) so that I receive an exact, cited solution grounded only in official documentation.
- **US-02 (Transparent Reasoning):** As an engineer, I want to inspect the retrieved chunks and confidence scores so that I can verify the authenticity of the generated answer.
- **US-03 (Graceful Fallback):** As an engineer, I want the system to explicitly state when documentation does not contain the answer, rather than inventing false instructions.
- **US-04 (Automated Regression Testing):** As an AI engineer maintaining the system, I want to run `make eval` to benchmark changes against a golden dataset before deploying.

---

## 6. Core Features

### Feature 1: Hybrid Dual-Channel Retrieval & Reciprocal Rank Fusion (RRF)
- **Purpose:** Combine semantic density (Dense vectors via BGE-M3) with exact lexical matching (Sparse BM25) to prevent missed error codes and keywords.
- **Inputs:** Raw user query string.
- **Outputs:** Top-15 unified candidates ranked by reciprocal rank fusion score.
- **Acceptance Criteria:**
  - Queries containing exact error codes (e.g., `502 Bad Gateway`) must rank the relevant doc in top-3 candidates.
  - Fusion execution latency must remain under 60ms.

### Feature 2: Cross-Encoder Reranking
- **Purpose:** Deep contextual scoring of the top-15 candidates to select the top-5 most relevant passages.
- **Inputs:** User query + Top-15 RRF candidates.
- **Outputs:** Top-5 rescored and sorted passages with relevance scores.
- **Acceptance Criteria:** Cross-Encoder (`BAAI/bge-reranker-v2-m3`) must score relevance from 0.0 to 1.0.

### Feature 3: Corrective RAG (CRAG) Quality Gate & Self-Correction
- **Purpose:** Evaluate whether the retrieved passages are sufficient to answer the query. If passages are ambiguous or irrelevant, rewrite the query or gracefully decline.
- **Inputs:** Query + Top-5 reranked passages.
- **Outputs:** Decision state (`CORRECT`, `AMBIGUOUS`, `INCORRECT`) + refined query if needed.
- **Acceptance Criteria:** If max relevance score is below 0.35, the system must trigger query rewrite or fallback response without hallucination.

### Feature 4: Local High-Throughput Generation via vLLM
- **Purpose:** Generate grounded responses with token streaming via Server-Sent Events (SSE).
- **Inputs:** Prompt with structured `<context>` tags + chat history.
- **Outputs:** Token-by-token streamed response in Chainlit UI.
- **Acceptance Criteria:** First-token latency (TTFT) < 350ms on local GPU.

### Feature 5: Offline Evaluation & CI Benchmark Pipeline (Ragas + DeepEval)
- **Purpose:** Measure quantitative retrieval and generation metrics across a golden dataset.
- **Inputs:** `data/golden_eval_dataset.json` (50 curated QA pairs).
- **Outputs:** Metric report (`Context Precision`, `Context Recall`, `Faithfulness`, `Answer Relevancy`).
- **Acceptance Criteria:** Automatically fail CI if `Faithfulness` drops below 0.85 or `Context Recall` drops below 0.80.

---

## 7. Non-Goals (Out of Scope for MVP)
- Multi-tenant cloud billing and subscription tiers.
- Real-time internet search crawling (focus is strictly on curated local documentation).
- GraphRAG multi-hop relational extraction (deferred to future roadmap).
- Mobile apps (iOS/Android) — responsive web interface is sufficient.

---

## 8. MVP Scope
### Must Have (P0)
- End-to-end Hybrid Search (BGE-M3 Dense + BM25 Sparse + RRF).
- BGE-Reranker-v2-m3 Cross-Encoder step.
- CRAG confidence grading node in LangGraph.
- Local inference connection to vLLM (OpenAI-compatible).
- Qdrant Vector DB integration.
- Automated Ragas benchmarking script producing markdown/JSON results.
- Chainlit interactive UI with step-by-step reasoning visibility.

### Nice to Have (P1)
- Redis semantic caching for repeated queries.
- Langfuse automated trace integration.
- Automated synthetic dataset generator using Ragas.

### Future Roadmap (P2)
- ColPali multimodal parsing for PDF architectural diagrams.
- Distributed agentic routing across multiple domain-specific vector stores.

---

## 9. Success Metrics
- **Context Recall@5:** $\ge 0.90$
- **Context Precision@5:** $\ge 0.85$
- **Faithfulness (Zero-hallucination):** $\ge 0.92$
- **Answer Relevancy:** $\ge 0.88$
- **P95 End-to-End Latency:** $\le 500	ext{ ms}$ (with local GPU inference)
- **API Cost per Query:** $\$0.00$ (100% open-source local serving)

---

## 10. Constraints
- **Hardware:** Must run smoothly on consumer NVIDIA GPUs (e.g., 12GB–24GB VRAM) using AWQ quantization.
- **Language:** Must natively support mixed Persian and English text without encoding corruption.
- **Infrastructure:** All services must be orchestrated via Docker Compose for one-command deployment.

---

## 11. Assumptions
- Documentation is provided in clean Markdown or text formats.
- The host system has Docker and NVIDIA Container Toolkit installed.
- Local model weights for Qwen 2.5 and BGE are cached locally.

---

## 12. Definition of Done (DoD)
- [ ] All 5 core pipeline components (Dense, Sparse, RRF, Rerank, CRAG) implemented and unit-tested.
- [ ] Docker Compose brings up Qdrant, Redis, Langfuse, and FastAPI without errors.
- [ ] Ragas evaluation run produces verified score report exceeding MVP thresholds.
- [ ] `README.md` documents architecture, benchmarks, and honest edge-case failure modes.
- [ ] Linting (`ruff`), typing (`mypy`), and test suite (`pytest`) pass with 0 errors.
