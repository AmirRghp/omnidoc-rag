# Using Profiles with Docker Compose

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
