# Building Multi-Platform Images with Docker Buildx

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
