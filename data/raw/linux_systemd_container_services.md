# Running Containers as Systemd Services

Systemd can manage container lifecycles to ensure automatic startup after reboot.

## Unit File Example

```ini
[Unit]
Description=Docker Postgres Production Container
Requires=docker.service
After=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker start -a prod-postgres
ExecStop=/usr/bin/docker stop -t 10 prod-postgres

[Install]
WantedBy=multi-user.target
```
