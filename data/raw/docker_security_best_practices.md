# Docker security best practices

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
