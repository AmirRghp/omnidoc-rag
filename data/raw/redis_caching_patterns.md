# Redis In-Memory Caching & Eviction Policies

Redis acts as high-throughput cache for query results and vector embeddings.

## Eviction Policies

When `maxmemory` is reached, Redis evicts keys according to policy:
- `allkeys-lru`: Evicts least recently used keys out of all keys.
- `volatile-lru`: Evicts least recently used keys with an expire set.
- `noeviction`: Returns an error on write commands when limit reached.

```ini
maxmemory 2gb
maxmemory-policy allkeys-lru
```
