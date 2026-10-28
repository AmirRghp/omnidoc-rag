# Troubleshoot the Docker daemon

Common troubleshooting procedures when the Docker daemon fails to start or respond.

## Daemon status check

Check daemon status on Linux:
```bash
systemctl status docker
journalctl -u docker.service --no-pager -n 50
```

## Socket Permission Denied

If you encounter:
```
permission denied while trying to connect to the Docker daemon socket
```
Ensure your user is part of the `docker` group:
```bash
sudo usermod -aG docker $USER
```
