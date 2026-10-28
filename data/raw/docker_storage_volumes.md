# Manage data in Docker

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
