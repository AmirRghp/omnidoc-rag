# docker run command reference

The `docker run` command creates a writeable container layer and starts it.

## Common Flags

- `--detach`, `-d`: Run container in background and print container ID.
- `--interactive`, `-i`: Keep STDIN open even if not attached.
- `--tty`, `-t`: Allocate a pseudo-TTY.
- `--restart`: Restart policy (`no`, `on-failure`, `always`, `unless-stopped`).
- `--ipc`: IPC mode to use, such as `host` or `private`.
- `--memory`: Memory limit.
- `--cpus`: Number of CPUs.
