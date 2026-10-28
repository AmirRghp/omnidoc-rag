# Linux cgroups Resource Management in Containers

Control groups isolate resource usage (CPU, memory, disk I/O, network).

## cgroups v2

Modern Linux systems use cgroups v2 (`/sys/fs/cgroup/`).
Docker maps `--memory`, `--cpus`, and `--pids-limit` directly to cgroups:
- `memory.max`: Hard limit on memory usage.
- `cpu.max`: Quota and period for CPU bandwidth.
- `pids.max`: Process thread limit to prevent fork bombs.
