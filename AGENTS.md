# AI Coding Agent Instructions (AGENTS.md)

## 1. Project Context & Tech Stack
- **Project:** OmniDoc RAG (Robust Bilingual Technical RAG & Evaluation System)
- **Primary Domain:** Technical infrastructure documentation (Docker, PostgreSQL, Linux) with mixed English/Persian queries.
- **Core Stack:** Python 3.11+, FastAPI, LangGraph, Qdrant, Redis, vLLM (local serving), BGE-M3, Ragas.
- **Mission:** Implement an enterprise-grade hybrid retrieval and evaluation system without regressions, untyped dictionaries, or hardcoded external APIs.

---

## 2. Source of Truth Hierarchy
Before implementing or modifying any module, read and conform to the documentation files in `docs/`:
1. `docs/PRD.md` — Core requirements and acceptance criteria.
2. `docs/ARCHITECTURE.md` — Layer boundaries, data flow, and ADR decisions.
3. `docs/DATABASE.md` — Vector collection schemas, payloads, and Redis keys.
4. `docs/API.md` — Request/response schemas and SSE streaming contracts.
5. `docs/SECURITY.md` — Strict context tagging, input sanitization, and secrets rules.
6. `docs/CODE_STYLE.md` — Strict Python typing, async rules, and error handling.
7. `docs/TESTING.md` — Pytest fixtures, mock behaviors, and Ragas gating thresholds.

*Rule: Never guess requirements or invent architectural shortcuts that contradict these documents.*

---

## 3. Local Skills Catalog (Progressive Disclosure)
All project skills are committed locally under `.agents/skills/`.

> [!IMPORTANT]
> **Progressive Disclosure Rule:** Never load or read all skills at session start. Agents must only inspect the specific `SKILL.md` required for the immediate task phase.

| Skill Name | Local Path | Trigger Condition |
|---|---|---|
| `context-engineering` | `.agents/skills/context-engineering/SKILL.md` | Session initialization, context window cleanup, prompt structuring |
| `interview-me` | `.agents/skills/interview-me/SKILL.md` | Ambiguous, underspecified requests; clarifying user intent before planning |
| `spec-driven-development` | `.agents/skills/spec-driven-development/SKILL.md` | Creating specifications and PRDs before writing any code |
| `planning-and-task-breakdown` | `.agents/skills/planning-and-task-breakdown/SKILL.md` | Decomposing specs into small, verifiable, ordered tasks |
| `graphify` | `.agents/skills/graphify/SKILL.md` | Exploring codebase structure, architecture relations, and GraphRAG knowledge graphs |
| `incremental-implementation` | `.agents/skills/incremental-implementation/SKILL.md` | Delivering features in thin vertical slices; multi-file edits |
| `source-driven-development` | `.agents/skills/source-driven-development/SKILL.md` | Grounding framework patterns in official documentation; anti-hallucination |
| `doubt-driven-development` | `.agents/skills/doubt-driven-development/SKILL.md` | Adversarial stress-testing of non-trivial architectural and security decisions |
| `constraint-driven-development` | `.agents/skills/constraint-driven-development/SKILL.md` | Establishing and enforcing non-negotiable quality and performance thresholds |
| `managing-python-dependencies` | `.agents/skills/managing-python-dependencies/SKILL.md` | Python virtualenv, uv/poetry/pip management; preventing global package pollution |
| `api-and-interface-design` | `.agents/skills/api-and-interface-design/SKILL.md` | FastAPI route contracts, Pydantic v2 schemas, SSE streaming endpoints |
| `frontend-ui-engineering` | `.agents/skills/frontend-ui-engineering/SKILL.md` | UI components, accessibility, state management, anti-AI aesthetic |
| `persian-rtl-ux` | `.agents/skills/persian-rtl-ux/SKILL.md` | Bilingual Persian/English text, RTL layout, mixed bidirectional numbers & code |
| `test-driven-development` | `.agents/skills/test-driven-development/SKILL.md` | Writing unit/integration tests first; Red-Green-Refactor loop |
| `debugging-and-error-recovery` | `.agents/skills/debugging-and-error-recovery/SKILL.md` | Systematic root-cause debugging; test failures, build errors |
| `accidental-data-loss-prevention` | `.agents/skills/accidental-data-loss-prevention/SKILL.md` | Halting and confirming before DROP, TRUNCATE, or destructive schema changes |
| `code-simplification` | `.agents/skills/code-simplification/SKILL.md` | Reducing complexity while strictly preserving behavior; removing clutter |
| `code-review-and-quality` | `.agents/skills/code-review-and-quality/SKILL.md` | 5-axis review (correctness, readability, architecture, security, performance) |
| `thermo-nuclear-code-quality-review` | `.agents/skills/thermo-nuclear-code-quality-review/SKILL.md` | Ultra-strict anti-bloat audit; eliminating shallow helpers, giant files, and spaghetti |
| `security-and-hardening` | `.agents/skills/security-and-hardening/SKILL.md` | STRIDE threat modeling, prompt injection defense, trust boundaries |
| `security-best-practices` | `.agents/skills/security-best-practices/SKILL.md` | Language/framework specific hardening (Python & FastAPI security guides) |
| `performance-optimization` | `.agents/skills/performance-optimization/SKILL.md` | Profiling bottlenecks, async CPU/IO offloading, vector search latency |
| `observability-and-instrumentation` | `.agents/skills/observability-and-instrumentation/SKILL.md` | Structured telemetry, logging, metrics, correlation IDs |
| `git-workflow-and-versioning` | `.agents/skills/git-workflow-and-versioning/SKILL.md` | Trunk-based workflow, atomic commits, branch naming, semantic versioning |
| `gh-fix-ci` | `.agents/skills/gh-fix-ci/SKILL.md` | Inspecting and resolving failing GitHub Actions CI checks via `gh` CLI |
| `gh-address-comments` | `.agents/skills/gh-address-comments/SKILL.md` | Triaging and resolving PR review comments via `gh` CLI |
| `ci-cd-and-automation` | `.agents/skills/ci-cd-and-automation/SKILL.md` | Setting up and maintaining CI pipelines and quality gates |
| `shipping-and-launch` | `.agents/skills/shipping-and-launch/SKILL.md` | Production readiness checklists, staged rollout, and rollback plans |
| `documentation-and-adrs` | `.agents/skills/documentation-and-adrs/SKILL.md` | Recording architectural decisions (ADRs) and documenting system trade-offs |

---

## 4. Task-Based Skill Routing & Workflows

Follow this lifecycle for all development tasks:

```
┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│ Clarification│ ──> │ Planning     │ ──> │ Implementation │ ──> │ Verification │ ──> │ Review & Git │
└──────────────┘     └──────────────┘     └────────────────┘     └──────────────┘     └──────────────┘
```

### 4.1 Discovery & Clarification
- When requirements are vague or underspecified: Activate `interview-me`. Ask one targeted question at a time until intent is ≥95% clear.
- When exploring codebase topology or complex dependencies: Activate `graphify`.

### 4.2 Planning & Architecture
- For new features or significant changes: Read `docs/ARCHITECTURE.md` and `docs/PRD.md`, then activate `spec-driven-development`.
- Break specifications into small, atomic tasks: Activate `planning-and-task-breakdown`.
- For non-trivial architectural, data model, or concurrency decisions: Activate `doubt-driven-development` to stress-test assumptions adversarially.
- Define quality constraints and metric floors: Activate `constraint-driven-development`.
- Document key trade-offs and decisions: Activate `documentation-and-adrs`.

### 4.3 Implementation Discipline
- Build in thin, vertical slices: Activate `incremental-implementation`. Never write large unverified code dumps.
- Verify library patterns against official docs: Activate `source-driven-development`. Never implement APIs from stale memory.
- For Python environments and package additions: Activate `managing-python-dependencies`. Never execute global `pip install`.
- For API endpoints and schema models: Activate `api-and-interface-design` with Pydantic v2 and FastAPI contracts.
- For UI components: Activate `frontend-ui-engineering`.
- For Persian or mixed-direction text: Activate `persian-rtl-ux`.
- For adding telemetry and metrics: Activate `observability-and-instrumentation`.

### 4.4 Testing, Debugging & Safety
- Before writing implementation code: Activate `test-driven-development` (write failing test first).
- When a test fails or error occurs: Activate `debugging-and-error-recovery` (Stop-the-line triage; diagnose root cause before resuming).
- When dealing with data or schema destruction: Activate `accidental-data-loss-prevention` (require affirmative confirmation).

### 4.5 Quality Review & Anti-Slop
- After implementation is functional: Activate `code-simplification` to delete unnecessary abstractions and clarify logic.
- Conduct standard pre-merge audit: Activate `code-review-and-quality`.
- For deep structural audits (files > 500 lines or complex branches): Activate `thermo-nuclear-code-quality-review`.
- Security validation: Activate `security-and-hardening` + `security-best-practices`.
- Performance profiling: Activate `performance-optimization`.

### 4.6 Collaboration & Delivery
- Version control and atomic commits: Activate `git-workflow-and-versioning`.
- Resolving GitHub PR comments: Activate `gh-address-comments`.
- Fixing GitHub CI build issues: Activate `gh-fix-ci`.
- Production readiness: Activate `shipping-and-launch`.

---

## 5. Skill Priority & Conflict Resolution Matrix

When multiple skills touch related areas, follow this explicit precedence order:

1. **Safety & Data Loss Precedence:**
   - `accidental-data-loss-prevention` > any optimization or migration script. Always stop and obtain confirmation before data-destructive operations.
2. **Architecture & Source Truth Precedence:**
   - `docs/ARCHITECTURE.md` + `source-driven-development` > training memory or generic LLM habits. Official documentation and project ADRs always override intuition.
3. **Planning Precedence:**
   - `interview-me` (intent) > `spec-driven-development` (spec) > `planning-and-task-breakdown` (tasks) > `incremental-implementation` (execution).
4. **Code Quality Precedence:**
   - `code-simplification` (clarity & behavior preservation) > `code-review-and-quality` (5-axis approval) > `thermo-nuclear-code-quality-review` (structural judo).
5. **Security Precedence:**
   - `security-and-hardening` (threat model & boundary control) takes precedence at design time; `security-best-practices` provides specific implementation details for Python/FastAPI.

---

## 6. Anti-Hallucination, Anti-AI Slop & Anti-Overengineering Guardrails

1. **Zero Hallucinated Dependencies:**
   - Consult `docs/ARCHITECTURE.md` and use `managing-python-dependencies`. Do NOT add third-party packages to `requirements.txt` or `pyproject.toml` without explicit user approval.
2. **Zero Unverified Code Patterns:**
   - Ground all FastAPI, LangGraph, Qdrant, and Redis usage in verified documentation (`source-driven-development`).
3. **Eliminate AI Slop:**
   - Reject verbose boilerplate, redundant docstrings stating the obvious, hollow helper functions, and generic AI aesthetic.
   - Follow `code-simplification`: if code does not carry its weight, remove it.
4. **Prevent Overengineering:**
   - Do NOT introduce abstract factory layers, redundant interfaces, or multi-tiered wrappers for simple operations.
   - Do NOT build speculative features or unrequested "future-proofing".
5. **Strict Async Boundary:**
   - Never block the event loop with synchronous file/network I/O. Use `asyncio.to_thread` for CPU-bound tasks (Cross-Encoder, BGE-M3 tokenization).

---

## 7. Mandatory Pre-Implementation Check
Before modifying or creating any file:
1. Locate and review the relevant specifications in `docs/`:
   - Route/SSE contracts -> `docs/API.md`
   - Data schemas & keys -> `docs/DATABASE.md`
   - Component boundaries -> `docs/ARCHITECTURE.md`
   - Quality & style rules -> `docs/CODE_STYLE.md`
2. Formulate a brief, explicit plan outlining modified files, signatures, and test cases.

---

## 8. Definition of Task Completion & Verification Protocol
A task is complete **ONLY** after all the following verification gates pass:
- [ ] Requirements from `docs/PRD.md` and spec are satisfied.
- [ ] Unit and integration tests pass with 0 failures (`pytest tests/`).
- [ ] Static type checker reports 0 errors (`mypy --strict`).
- [ ] Linter reports 0 errors (`ruff check .`).
- [ ] No unhandled assumptions or regressions exist (`doubt-driven-development`).
- [ ] No temporary debugging files or extraneous diffs remain.
- [ ] Documentation / ADR updated if interface contracts or schemas changed (`documentation-and-adrs`).
