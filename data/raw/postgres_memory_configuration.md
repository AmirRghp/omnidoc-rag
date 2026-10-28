# PostgreSQL Memory Configuration Guide

Optimal resource allocation prevents Out-Of-Memory (OOM) kills.

## Key Memory Parameters

- `shared_buffers`: Memory dedicated to caching data (typically 25% of RAM).
- `work_mem`: Memory used by query operations (sorts, hash tables) before disk.
- `maintenance_work_mem`: Memory used by `VACUUM`, `CREATE INDEX`, `ALTER TABLE`.
