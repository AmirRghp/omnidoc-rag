# Task 4.3: Persistent BM25 Lexical Indexer

Build and persist a bilingual (Persian/English) BM25 lexical inverted index (`BM25Okapi`) that preserves technical CLI flags, error codes, and Persian morphology, persisted atomically to disk for hybrid retrieval.

## User Review & Critical Decisions

> [!IMPORTANT]
> Review the proposed design choices below before proceeding to implementation:

- **Bilingual Technical Tokenizer**: Uses Unicode normalization (`normalize_persian_text` and `strip_directional_overrides` from `src.common.security`) and a regex pattern that preserves CLI flags (e.g., `--ipc=host`, `-p`), paths/identifiers (`max_connections`, `/var/run/docker.sock`), and Persian words containing Zero-Width Non-Joiners (`\u200c`).
- **Atomic File Replacement**: Writes serialized index state (`chunks`, `tokenized_corpus`, and `BM25Okapi`) to a sibling temporary file in `data/processed/` with `os.fsync` followed by `os.replace` to `data/processed/bm25_index.pkl` to guarantee zero partial-read corruption.
- **Score Calibration for `RetrievedChunk`**: Because `RetrievedChunk.score` enforces `[0.0, 1.0]`, positive raw BM25 scores are normalized to `[0.0, 1.0]` (via max-score scaling or $x / (x + 1)$ saturation) while preserving `"raw_bm25_score"` and `"bm25_rank"` in `RetrievedChunk.metadata`.

## 1. Overview & Core Concept

- **What It Does**: Tokenizes bilingual technical chunks, builds an in-memory `BM25Okapi` index, persists it atomically to `data/processed/bm25_index.pkl`, and supports fast lexical search and idempotent chunk updates.
- **Target Audience / Persona**: Downstream ingestion pipeline (`Task 4.4: IngestionPipeline`) and sparse retrieval engine (`Task 5.1: SparseRetriever`).
- **Key Value**: Captures exact lexical matches on CLI flags, config keys, error codes, and Persian technical terms where dense embeddings alone may underperform.

## 2. User Experience & Developer Workflow

- **Ingestion Flow**:
  1. Receive validated `list[DocumentChunkPayload]` chunks.
  2. Deduplicate by `point_id` (supporting both full index build and incremental `add_chunks` / `build_index`).
  3. Tokenize each chunk's content (and section heading context) with `tokenize_bilingual`.
  4. Construct `BM25Okapi` and atomically persist to `data/processed/bm25_index.pkl`.
- **Retrieval & Persistence Flow**:
  1. Load existing index from `data/processed/bm25_index.pkl` via `load()`.
  2. Query with bilingual technical strings and receive ranked `RetrievedChunk` instances.

## 3. Key Product Decisions & Trade-Offs

- **Decision 1: Regex-based Bilingual Technical Tokenizer**
  - *Chosen Approach*: Normalize Arabic/Persian characters (`ی`, `ک`, preserve `\u200c`), lowercase ASCII, and extract tokens matching technical flags (`--?[a-zA-Z0-9][\w\-=./:]*`) or word/identifier sequences (`[\w\u0600-\u06FF\u200c]+(?:[./_\-:][\w\u0600-\u06FF\u200c]+)*`).
  - *Why*: Standard whitespace or punctuation splitters strip `--` from `--ipc=host` or break `max_connections` and Persian ZWNJ compound words.
  - *Alternatives Considered*: `tiktoken` subword tokens (splits domain identifiers arbitrarily and harms BM25 IDF specificity).
- **Decision 2: Atomic Temp-File + `os.replace` Persistence**
  - *Chosen Approach*: Write to a `NamedTemporaryFile` in the same directory as `bm25_index.pkl`, flush + `os.fsync`, and call `os.replace`.
  - *Why*: POSIX `os.replace` on the same filesystem is an atomic rename, preventing concurrent readers from observing a truncated pickle file.

## 4. Technical Architecture & Data Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                   src/ingestion/bm25_store.py                  │
├────────────────────────────────────────────────────────────────┤
│  tokenize_bilingual(text: str) -> list[str]                    │
│    ├── strip_directional_overrides() & normalize_persian_text()│
│    └── Regex extraction (CLI flags, identifiers, FA/EN words)  │
├────────────────────────────────────────────────────────────────┤
│  BM25StoreManager                                              │
│    ├── build_index(chunks: list[DocumentChunkPayload]) -> int  │
│    ├── upsert_chunks(chunks: list[DocumentChunkPayload]) -> int│
│    ├── search(query: str, top_k: int) -> list[RetrievedChunk]  │
│    ├── asearch(query: str, top_k: int) -> list[RetrievedChunk] │
│    ├── save() -> Path  (TempFile + fsync + os.replace)         │
│    └── load() -> bool  (Loads from data/processed/bm25_index.pkl)│
└────────────────────────────────────────────────────────────────┘
```

### Verification Criteria
- `pytest tests/unit/test_bm25_store.py -v` passes with 0 failures.
- `ruff check src tests` and `ruff format --check src tests` pass with 0 errors.
- `mypy src` passes under strict mode with 0 errors.
