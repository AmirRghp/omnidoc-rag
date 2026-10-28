# Nginx Reverse Proxy Configuration for Docker Services

Configuring Nginx as an edge proxy with SSL termination and connection keepalive.

## Upstream Configuration

```nginx
upstream backend_cluster {
    server 172.18.0.2:8000 max_fails=3 fail_timeout=10s;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://backend_cluster;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
    }
}
```
