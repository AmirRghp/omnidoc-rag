# Table Partitioning in PostgreSQL

Partitioning splits one large logical table into smaller physical pieces.

## Range Partitioning Example

```sql
CREATE TABLE measurement (
    city_id         int not null,
    logdate         date not null,
    peaktemp        int
) PARTITION BY RANGE (logdate);

CREATE TABLE measurement_y2026m10 PARTITION OF measurement
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
```
Query performance improves when scanning only relevant partitions.
