# Routine Vacuuming and Table Bloat

PostgreSQL's MVCC creates new row versions on `UPDATE` and `DELETE`.

## Autovacuum Tuning

Tune autovacuum settings to keep pace with high-write workloads:
```ini
autovacuum = on
autovacuum_max_workers = 4
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02
```
Check bloat and dead tuples using `pg_stat_user_tables`.
