# PostgreSQL Architecture Overview

PostgreSQL is a powerful, open-source object-relational database system.

## Process Architecture

PostgreSQL runs as a set of cooperating processes:
- `postmaster`: listens for connections and forks backend processes.
- Background writer: flushes dirty shared buffers to disk.
- Checkpointer: executes periodic checkpoints.
- WAL writer: writes Write-Ahead Log data from shared memory to disk.
- Autovacuum launcher: reclaims storage by removing dead row versions.
