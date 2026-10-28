# Healthcheck configuration in Docker

The `HEALTHCHECK` instruction tells Docker how to test container health.

## Dockerfile Syntax

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
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
