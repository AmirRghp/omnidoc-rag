# Database & Storage Specification (DATABASE.md)

## 1. Storage Architecture Overview
OmniDoc RAG utilizes a specialized multi-storage architecture:
1. **Qdrant (Vector Database):** Primary store for document chunks, 1024-dimensional dense vectors, and structured payload metadata.
2. **Redis 7 (In-Memory Key-Value Store):** Semantic query caching, rate limiting, and temporary conversation session buffers.
3. **Local Serialized Inverted Index:** Disk-backed pickle/JSON index for BM25 term frequencies.
4. **PostgreSQL 16 (Observability & Operational Store):** Dedicated relational database used internally by Langfuse to store traces, spans, evaluations, and user feedback.

---

## 2. Naming Conventions
- **Collections & Tables:** Lowercase snake_case (e.g., `tech_docs_bilingual`, `eval_runs`).
- **Vector Names:** Lowercase snake_case (e.g., `dense_vector`).
- **Keys in Redis:** Colon-separated namespaces: `<domain>:<entity>:<identifier>` (e.g., `cache:query:<sha256>`, `ratelimit:ip:<ip_address>`).
- **Payload Fields:** Lowercase snake_case (e.g., `doc_id`, `chunk_id`, `source_file`, `created_at`).
- **Timestamps:** ISO 8601 UTC strings (`YYYY-MM-DDTHH:MM:SSZ`).

---

## 3. Storage Definitions

### 3.1 Qdrant Vector Collection: `tech_docs_bilingual`

#### Collection Configuration:
```json
{
  "vectors": {
    "dense": {
      "size": 1024,
      "distance": "Cosine",
      "on_disk": false
    }
  },
  "optimizers_config": {
    "default_segment_number": 2,
    "indexing_threshold": 10000
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 1000
  }
}
```

#### Point Payload Schema:
| Field Name | Type | Indexed | Description |
| :--- | :--- | :--- | :--- |
| `point_id` | UUIDv4 (String) | Primary Key | Unique point identifier. |
| `doc_id` | String | Keyword Index | Identifier of parent document. |
| `chunk_id` | Integer | Integer Index | 0-indexed sequential chunk index within document. |
| `content` | String | Full-text (Optional) | Raw text content of the chunk (UTF-8). |
| `source_file` | String | Keyword Index | Relative file path (e.g., `manuals/docker_daemon.md`). |
| `section_heading` | String | Keyword Index | Nearest Markdown header (e.g., `## Daemon socket option`). |
| `language` | String (`"fa"` / `"en"`) | Keyword Index | Primary language detected in chunk. |
| `token_count` | Integer | Integer Index | Token count using cl100k_base / Qwen tokenizer. |
| `created_at` | String (ISO 8601) | DateTime Index | Ingestion timestamp. |

---

### 3.2 Redis Key Schema

| Key Pattern | Data Type | TTL | Purpose |
| :--- | :--- | :--- | :--- |
| `cache:query:<sha256_hash>` | String (JSON) | 86400s (24h) | Caches verified answer, citations, and model version for identical normalized queries. |
| `ratelimit:ip:<client_ip>` | Integer (Counter) | 60s (1 min) | Sliding window rate limit counter (max 30 requests/minute). |
| `session:<session_id>:history` | List (JSON) | 3600s (1h) | Stores last 6 turns of user conversation for multi-turn context resolution. |

---

### 3.3 Relational Schema (PostgreSQL for Evaluations & Logging)

#### Table: `eval_runs`
```sql
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name VARCHAR(100) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    retrieval_strategy VARCHAR(50) NOT NULL, -- 'dense_only', 'hybrid', 'hybrid_rerank'
    context_recall NUMERIC(5, 4) NOT NULL,
    context_precision NUMERIC(5, 4) NOT NULL,
    faithfulness NUMERIC(5, 4) NOT NULL,
    answer_relevancy NUMERIC(5, 4) NOT NULL,
    p95_latency_ms INTEGER NOT NULL,
    total_samples INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## 4. Constraints & Data Integrity
- **Deduplication:** Before inserting a document chunk, compute MD5 hash of `source_file + ":" + str(chunk_id)`. Upserting with deterministic UUID based on this hash guarantees idempotency.
- **Payload Validation:** Enforced strictly via Pydantic model `DocumentChunkPayload` before writing to Qdrant.
- **Encoding:** Strict UTF-8 enforcement to prevent corruption of Persian characters (e.g., `ی` and `ک` normalization).

---

## 5. Backup & Recovery Strategy
- **Qdrant Snapshots:** Periodic snapshot trigger via Qdrant REST API (`POST /collections/tech_docs_bilingual/snapshots`) saved to persistent volume `/qdrant/snapshots`.
- **BM25 Persistence:** Pickled model checkpoint `data/processed/bm25_index.pkl` persisted with atomic file replacement (`os.replace`) to prevent read corruptions during indexing.
