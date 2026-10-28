# docker build command reference

The `docker build` command builds an image from a Dockerfile and a build context.

## Build Context

The build context is the set of files located at the specified `PATH` or `URL`.

```bash
docker build -t my-app:1.0.0 -f Dockerfile .
```

### Multi-stage builds
Multi-stage builds allow you to drastically reduce image sizes cleanly.
