# Use host networking

If you use `host` network mode, the container's network stack is not isolated.
The container does not get its own IP-address allocated.

## Usage

Pass `--network=host` or `--net=host` to the `docker run` command:
```bash
docker run --rm -d --network=host --name my_nginx nginx
```

### Port Mapping Limitations
Port mapping with `-p` is ignored because container binds to host interfaces directly.
