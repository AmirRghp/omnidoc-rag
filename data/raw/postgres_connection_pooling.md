# Connection Management and Pooling in PostgreSQL

PostgreSQL allocates dedicated memory and processes for each client connection.

## Connection Limits

Configure `max_connections` in `postgresql.conf`:
```ini
max_connections = 200
```

## Connection Pool Exhaustion

When connections run out, clients observe:
```
ERR_DB_TIMEOUT: connection pool exhausted
FATAL: remaining connection slots are reserved for non-replication superuser connections
```

Use lightweight poolers such as PgBouncer in transaction pooling mode.
