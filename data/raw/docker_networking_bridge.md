# Use bridge networks

A bridge network is a Link Layer device that forwards traffic between network segments.
A bridge can be a hardware device or a software device running in the host kernel.

## User-defined bridge networks

User-defined bridge networks allow containers to communicate on the same host.

```bash
docker network create my-net
docker run -d --name my-nginx --network my-net nginx:alpine
```

### DNS resolution
Containers on a user-defined bridge network resolve each other by container name.
Containers on the default bridge network do not have automatic DNS resolution.
