# Configure Logging Drivers in Docker

Docker includes multiple logging mechanisms to manage container output.

## Supported Drivers

- `json-file`: Logs formatted as JSON (default driver).
- `syslog`: Writes messages to syslog.
- `journald`: Writes log messages to systemd-journald.

Configure limits in `/etc/docker/daemon.json`:
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```
