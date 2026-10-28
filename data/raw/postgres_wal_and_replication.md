# Write-Ahead Logging (WAL) and Streaming Replication

PostgreSQL writes modifications to WAL before changes are written to data pages.

## Replication Settings

```ini
wal_level = replica
max_wal_size = 16GB
min_wal_size = 1GB
archive_mode = on
archive_command = 'cp %p /mnt/wal_archive/%f'
```
Standby servers stay up-to-date by streaming WAL records directly.
