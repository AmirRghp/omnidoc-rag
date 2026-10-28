# Docker Engine Overview

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
