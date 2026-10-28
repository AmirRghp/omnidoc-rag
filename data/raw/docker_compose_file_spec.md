# Compose file reference

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
