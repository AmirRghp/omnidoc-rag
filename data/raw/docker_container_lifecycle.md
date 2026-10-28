# Docker Container Lifecycle States

A container progresses through several distinct states during its lifecycle.

## States

1. `Created`: Container created via `docker create` but not started.
2. `Running`: Primary process inside the container is actively executing.
3. `Paused`: All processes in the container are suspended via cgroups.
4. `Restarting`: Container exited and daemon is restarting it.
5. `Exited`: Container process terminated with exit code `0` or an error.
6. `Dead`: Daemon failed to cleanly stop or unmount resources.
