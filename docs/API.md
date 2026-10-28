# API Contracts & Specification (API.md)

## 1. Overview
The OmniDoc RAG API provides endpoints for real-time technical querying, streaming generation, document ingestion, and evaluation benchmarking.
- **Base URL:** `http://localhost:8080/api/v1`
- **Protocol:** HTTP/1.1 and HTTP/2
- **Data Format:** JSON (`application/json`) and Server-Sent Events (`text/event-stream`)
- **Authentication:** Bearer API Token (`Authorization: Bearer <API_KEY>`)

---

## 2. Endpoints

### 2.1 Query Execution (Streaming via SSE)
`POST /api/v1/query/stream`

Streams the generation token-by-token along with intermediate event payloads (retrieval steps, reranking scores, and final citations).

#### Request Headers:
```http
Content-Type: application/json
Accept: text/event-stream
Authorization: Bearer <API_KEY>
```

#### Request Body:
```json
{
  "query": "چگونه می‌توان پورت‌های کانتینر داکر را به صورت داینامیک نگاشت کرد؟",
  "session_id": "sess_8f3a9e12-78d1-49b8",
  "temperature": 0.2,
  "enable_reflection": true
}
```

#### Stream Events Sequence:
1. **Event: `step_start`** (Retrieval initiated)
   ```json
   { "event": "step_start", "step": "hybrid_retrieval", "timestamp": "2026-10-28T00:15:01Z" }
   ```
2. **Event: `retrieval_complete`** (Returns top reranked chunks)
   ```json
   {
     "event": "retrieval_complete",
     "count": 5,
     "top_score": 0.892,
     "sources": ["manuals/docker_networking.md", "manuals/docker_run.md"]
   }
   ```
3. **Event: `token`** (Streamed generation chunk)
   ```json
   { "event": "token", "content": "برای" }
   { "event": "token", "content": " نگاشت" }
   { "event": "token", "content": " پورت‌ها..." }
   ```
4. **Event: `done`** (Completed with full citations and trace metadata)
   ```json
   {
     "event": "done",
     "trace_id": "tr_6a81d02c",
     "latency_ms": 420,
     "tokens_used": 184
   }
   ```

---

### 2.2 Synchronous Query
`POST /api/v1/query`

Synchronous endpoint returning the complete grounded answer and citation metadata in a single JSON payload.

#### Response Body (`200 OK`):
```json
{
  "query": "چگونه می‌توان پورت‌های کانتینر داکر را به صورت داینامیک نگاشت کرد؟",
  "answer": "برای نگاشت داینامیک پورت‌ها در داکر می‌توانید از فلگ `-P` یا `--publish-all` استفاده کنید...",
  "citations": [
    {
      "source_file": "manuals/docker_run.md",
      "section_heading": "## Exposing and publishing ports",
      "relevance_score": 0.892,
      "snippet": "Use the -P or --publish-all flag to publish all exposed ports to random ports on the host interfaces."
    }
  ],
  "reflection_status": "CORRECT",
  "latency_ms": 412,
  "trace_id": "tr_6a81d02c"
}
```

---

### 2.3 Document Ingestion
`POST /api/v1/ingest`

Upload and trigger background indexing of new technical markdown documents.

#### Request Body (`multipart/form-data`):
- `file`: Binary file upload (`.md` or `.pdf`, max 25MB).
- `metadata`: Optional JSON string containing document tags or category.

#### Response Body (`202 Accepted`):
```json
{
  "job_id": "ingest_9a12c4b8-2026",
  "filename": "postgresql_connection_pooling.md",
  "status": "PROCESSING",
  "message": "File received and queued for semantic chunking and dual-indexing."
}
```

---

### 2.4 Ingestion Job Status
`GET /api/v1/ingest/status/{job_id}`

#### Response Body (`200 OK`):
```json
{
  "job_id": "ingest_9a12c4b8-2026",
  "status": "COMPLETED",
  "chunks_created": 34,
  "duration_seconds": 2.14,
  "error": null
}
```

---

### 2.5 Run Evaluation Benchmark
`POST /api/v1/eval/run`

Triggers an offline evaluation run over the golden benchmark dataset.

#### Request Body:
```json
{
  "sample_limit": 50,
  "compare_with_baseline": true
}
```

#### Response Body (`200 OK`):
```json
{
  "run_id": "eval_20261028_001",
  "metrics": {
    "context_recall": 0.912,
    "context_precision": 0.864,
    "faithfulness": 0.941,
    "answer_relevancy": 0.895
  },
  "p95_latency_ms": 435,
  "pass_threshold": true
}
```

---

## 3. Error Handling Contract (RFC 7807)
All error responses adhere to standard Problem Details formatting:

```json
{
  "type": "https://api.omnidoc.local/errors/VLLM_UNAVAILABLE",
  "title": "Inference Engine Unreachable",
  "status": 503,
  "detail": "Failed to connect to local vLLM server at http://localhost:8000/v1. Ensure vLLM container is active.",
  "instance": "/api/v1/query/stream",
  "code": "ERR_VLLM_CONNECTION_FAILED",
  "timestamp": "2026-10-28T00:15:30Z"
}
```
