"""Automated documentation fetcher and seed generator for technical Markdown manuals.

Fetches official Docker/PostgreSQL markdown reference files or seeds a robust offline
technical documentation catalog under data/raw/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

import structlog

logger = structlog.get_logger(__name__)

OFFLINE_FALLBACK_DOCS: Final[dict[str, str]] = {
    "docker_overview.md": """# Docker Engine Overview

Docker is an open platform for developing, shipping, and running applications.
Docker enables you to separate your applications from your infrastructure.

## Architecture

Docker uses a client-server architecture. The Docker client talks to the Docker daemon (`dockerd`),
which does the heavy lifting of building, running, and distributing your containers.

### Docker Daemon
The Docker daemon listens for Docker API requests and manages Docker objects.
A daemon can also communicate with other daemons to manage Docker services.

### Docker Client
The Docker client (`docker`) is the primary way that users interact with Docker.
When you use commands like `docker run`, the client sends commands to `dockerd`.
""",
    "docker_storage_volumes.md": """# Manage data in Docker

Volumes are the preferred mechanism for persisting data in Docker containers.
While bind mounts depend on the host OS structure, volumes are managed by Docker.

## Advantages of volumes

- Volumes are easier to back up or migrate than bind mounts.
- You can manage volumes using Docker CLI commands or the Docker API.
- Volumes work on both Linux and Windows containers.
- Volumes can be more safely shared among multiple containers.

## Use a volume

To create a volume:
```bash
docker volume create my-vol
```

To run a container with a volume mounted:
```bash
docker run -d --name devtest -v my-vol:/app nginx:latest
```
""",
    "docker_networking_bridge.md": """# Use bridge networks

A bridge network is a Link Layer device that forwards traffic between network segments.
A bridge can be a hardware device or a software device running in the host kernel.

## User-defined bridge networks

User-defined bridge networks allow containers to communicate on the same host.

```bash
docker network create my-net
docker run -d --name my-nginx --network my-net nginx:alpine
```

### DNS resolution
Containers on a user-defined bridge network resolve each other by container name.
Containers on the default bridge network do not have automatic DNS resolution.
""",
    "docker_networking_host.md": """# Use host networking

If you use `host` network mode, the container's network stack is not isolated.
The container does not get its own IP-address allocated.

## Usage

Pass `--network=host` or `--net=host` to the `docker run` command:
```bash
docker run --rm -d --network=host --name my_nginx nginx
```

### Port Mapping Limitations
Port mapping with `-p` is ignored because container binds to host interfaces directly.
""",
    "docker_cli_run.md": """# docker run command reference

The `docker run` command creates a writeable container layer and starts it.

## Common Flags

- `--detach`, `-d`: Run container in background and print container ID.
- `--interactive`, `-i`: Keep STDIN open even if not attached.
- `--tty`, `-t`: Allocate a pseudo-TTY.
- `--restart`: Restart policy (`no`, `on-failure`, `always`, `unless-stopped`).
- `--ipc`: IPC mode to use, such as `host` or `private`.
- `--memory`: Memory limit.
- `--cpus`: Number of CPUs.
""",
    "docker_cli_build.md": """# docker build command reference

The `docker build` command builds an image from a Dockerfile and a build context.

## Build Context

The build context is the set of files located at the specified `PATH` or `URL`.

```bash
docker build -t my-app:1.0.0 -f Dockerfile .
```

### Multi-stage builds
Multi-stage builds allow you to drastically reduce image sizes cleanly.
""",
    "docker_compose_file_spec.md": """# Compose file reference

The Compose file is a YAML file defining services, networks, and volumes.

## Top-level elements

- `services`: Defines the computing components of the application.
- `networks`: Defines shared communication networks across containers.
- `volumes`: Declares persistent storage volumes.
- `configs`: Adapts container behavior without needing to build images.
- `secrets`: Manages sensitive configuration data.

```yaml
version: '3.8'
services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    restart: always
```
""",
    "docker_troubleshooting_daemon.md": """# Troubleshoot the Docker daemon

Common troubleshooting procedures when the Docker daemon fails to start or respond.

## Daemon status check

Check daemon status on Linux:
```bash
systemctl status docker
journalctl -u docker.service --no-pager -n 50
```

## Socket Permission Denied

If you encounter:
```
permission denied while trying to connect to the Docker daemon socket
```
Ensure your user is part of the `docker` group:
```bash
sudo usermod -aG docker $USER
```
""",
    "docker_security_best_practices.md": """# Docker security best practices

Guidelines for hardening Docker containers and container host systems.

## Run as non-root user
Always define a non-root `USER` in your Dockerfile to mitigate container escape risks:
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

## Read-only root filesystem
Run containers with a read-only root filesystem when possible:
```bash
docker run --read-only nginx
```
""",
    "docker_healthchecks.md": """# Healthcheck configuration in Docker

The `HEALTHCHECK` instruction tells Docker how to test container health.

## Dockerfile Syntax

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \\
  CMD curl -f http://localhost:8080/health || exit 1
```

## Docker Compose Healthcheck

```yaml
services:
  web:
    image: my-web-app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```
""",
    "postgres_overview.md": """# PostgreSQL Architecture Overview

PostgreSQL is a powerful, open-source object-relational database system.

## Process Architecture

PostgreSQL runs as a set of cooperating processes:
- `postmaster`: listens for connections and forks backend processes.
- Background writer: flushes dirty shared buffers to disk.
- Checkpointer: executes periodic checkpoints.
- WAL writer: writes Write-Ahead Log data from shared memory to disk.
- Autovacuum launcher: reclaims storage by removing dead row versions.
""",
    "postgres_connection_pooling.md": """# Connection Management and Pooling in PostgreSQL

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
""",
    "postgres_memory_configuration.md": """# PostgreSQL Memory Configuration Guide

Optimal resource allocation prevents Out-Of-Memory (OOM) kills.

## Key Memory Parameters

- `shared_buffers`: Memory dedicated to caching data (typically 25% of RAM).
- `work_mem`: Memory used by query operations (sorts, hash tables) before disk.
- `maintenance_work_mem`: Memory used by `VACUUM`, `CREATE INDEX`, `ALTER TABLE`.
""",
    "postgres_vacuuming_and_bloat.md": """# Routine Vacuuming and Table Bloat

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
""",
    "postgres_wal_and_replication.md": """# Write-Ahead Logging (WAL) and Streaming Replication

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
""",
    "postgres_troubleshooting_locks.md": """# Troubleshooting Locks and Blocked Queries

Heavy concurrent DDL or unindexed foreign keys can cause lock contention.

## Lock Detection Query

```sql
SELECT
  blocked_locks.pid AS blocked_pid,
  blocked_activity.usename AS blocked_user,
  blocking_locks.pid AS blocking_pid,
  blocking_activity.usename AS blocking_user,
  blocked_activity.query AS blocked_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
    AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```
""",
    "linux_cgroups_resource_management.md": """# Linux cgroups Resource Management in Containers

Control groups isolate resource usage (CPU, memory, disk I/O, network).

## cgroups v2

Modern Linux systems use cgroups v2 (`/sys/fs/cgroup/`).
Docker maps `--memory`, `--cpus`, and `--pids-limit` directly to cgroups:
- `memory.max`: Hard limit on memory usage.
- `cpu.max`: Quota and period for CPU bandwidth.
- `pids.max`: Process thread limit to prevent fork bombs.
""",
    "linux_systemd_container_services.md": """# Running Containers as Systemd Services

Systemd can manage container lifecycles to ensure automatic startup after reboot.

## Unit File Example

```ini
[Unit]
Description=Docker Postgres Production Container
Requires=docker.service
After=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a prod-postgres
ExecStop=/usr/bin/docker stop -t 10 prod-postgres

[Install]
WantedBy=multi-user.target
```
""",
    "nginx_reverse_proxy_docker.md": """# Nginx Reverse Proxy Configuration for Docker Services

Configuring Nginx as an edge proxy with SSL termination and connection keepalive.

## Upstream Configuration

```nginx
upstream backend_cluster {
    server 172.18.0.2:8000 max_fails=3 fail_timeout=10s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://backend_cluster;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
    }
}
```
""",
    "redis_caching_patterns.md": """# Redis In-Memory Caching & Eviction Policies

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
""",
    "docker_multi_arch_buildx.md": """# Building Multi-Platform Images with Docker Buildx

Docker Buildx extends `docker` with the support of Moby BuildKit builder.

## Usage

Create a new builder instance:
```bash
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
```

Build and push for `linux/amd64` and `linux/arm64`:
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t myrepo/myapp:latest --push .
```
""",
    "docker_logging_drivers.md": """# Configure Logging Drivers in Docker

Docker includes multiple logging mechanisms to manage container output.

## Supported Drivers

- `json-file`: Logs formatted as JSON (default driver).
- `syslog`: Writes messages to syslog.
- `journald`: Writes log messages to systemd-journald.

Configure limits in `/etc/docker/daemon.json`:
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```
""",
    "postgres_partitioning_strategies.md": """# Table Partitioning in PostgreSQL

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
""",
    "docker_compose_profiles.md": """# Using Profiles with Docker Compose

Profiles adjust Compose application models for different environments.

## Profile Definition

```yaml
services:
  web:
    image: webapp

  backend:
    image: backend

  db-migration:
    image: migration-tool
    profiles: ["tools"]
```

Run only core services:
```bash
docker compose up
```
""",
    "docker_container_lifecycle.md": """# Docker Container Lifecycle States

A container progresses through several distinct states during its lifecycle.

## States

1. `Created`: Container created via `docker create` but not started.
2. `Running`: Primary process inside the container is actively executing.
3. `Paused`: All processes in the container are suspended via cgroups.
4. `Restarting`: Container exited and daemon is restarting it.
5. `Exited`: Container process terminated with exit code `0` or an error.
6. `Dead`: Daemon failed to cleanly stop or unmount resources.
""",
}


def fetch_from_github_api(
    repo: str,
    path: str,
    target_dir: Path,
    max_files: int = 30,
    github_token: str | None = None,
) -> list[Path]:
    """Attempt to fetch markdown files from GitHub repository via API.

    Args:
        repo: Repository identifier, e.g. 'docker/docs'.
        path: Directory path inside the repository, e.g. 'content/manuals'.
        target_dir: Destination folder on local filesystem.
        max_files: Maximum number of markdown files to download.
        github_token: Optional GitHub personal access token for higher rate limits.

    Returns:
        List of paths to downloaded Markdown files.
    """
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "User-Agent": "OmniDoc-RAG-Fetcher/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    req = urllib.request.Request(api_url, headers=headers)
    logger.info("Fetching repository tree from GitHub API", url=api_url, repo=repo)

    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, list):
        logger.warning("Unexpected GitHub API payload format", payload_type=type(payload).__name__)
        return []

    target_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    for item in payload:
        if len(saved_files) >= max_files:
            break
        if not isinstance(item, dict):
            continue
        if item.get("type") == "file" and str(item.get("name", "")).endswith(".md"):
            download_url = item.get("download_url")
            file_name = item.get("name")
            if not download_url or not file_name:
                continue

            file_req = urllib.request.Request(
                download_url,
                headers={"User-Agent": "OmniDoc-RAG-Fetcher/1.0"},
            )
            with urllib.request.urlopen(file_req, timeout=10) as file_resp:
                content = file_resp.read().decode("utf-8")

            dest_path = target_dir / file_name
            dest_path.write_text(content, encoding="utf-8")
            saved_files.append(dest_path)
            logger.info("Downloaded file from GitHub", path=str(dest_path))

    return saved_files


def seed_offline_documentation(target_dir: Path) -> list[Path]:
    """Seed comprehensive offline technical documentation catalog into target_dir.

    Args:
        target_dir: Destination folder.

    Returns:
        List of generated file paths.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    seeded_files: list[Path] = []

    for filename, content in OFFLINE_FALLBACK_DOCS.items():
        file_path = target_dir / filename
        file_path.write_text(content.strip() + "\n", encoding="utf-8")
        seeded_files.append(file_path)

    logger.info("Seeded offline technical documentation catalog", count=len(seeded_files))
    return seeded_files


def populate_documentation(
    target_dir: Path,
    max_files: int = 30,
    force_offline: bool = False,
    github_token: str | None = None,
) -> list[Path]:
    """Ensure target directory is populated with official technical documentation.

    Attempts GitHub API fetch first (unless force_offline=True), and automatically
    falls back to seeding the curated offline catalog on rate limit or network error.

    Args:
        target_dir: Target directory for raw files.
        max_files: Target file count limit.
        force_offline: Skip network requests and use offline catalog immediately.
        github_token: Optional token for GitHub API.

    Returns:
        List of all files present/created.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    if not force_offline:
        try:
            downloaded = fetch_from_github_api(
                repo="docker/docs",
                path="content/manuals/engine/install",
                target_dir=target_dir,
                max_files=max_files,
                github_token=github_token,
            )
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as err:
            logger.warning(
                "GitHub API fetch skipped or failed; applying offline seeding fallback",
                error=str(err),
            )

    if len(downloaded) < 10:
        # Fallback to local curated technical documentation suite
        seeded = seed_offline_documentation(target_dir)
        downloaded.extend(seeded)

    return downloaded


def main() -> int:
    """CLI Entry point for documentation fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch and seed raw technical documentation for OmniDoc RAG.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target directory to store Markdown files (default: data/raw)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=30,
        help="Maximum files to fetch (default: 30)",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip external network requests and use offline documentation seed catalog",
    )

    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN")

    logger.info(
        "Starting documentation acquisition",
        target_dir=str(args.target_dir),
        offline=args.offline_only,
    )
    files = populate_documentation(
        target_dir=args.target_dir,
        max_files=args.max_files,
        force_offline=args.offline_only,
        github_token=token,
    )
    logger.info("Documentation acquisition completed successfully", total_files=len(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
