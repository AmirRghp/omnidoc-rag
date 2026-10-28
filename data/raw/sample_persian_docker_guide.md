# راهنمای جامع عیب‌یابی و پیکربندی پیشرفته Docker و PostgreSQL (Bilingual Guide)

این راهنما شامل نکات کلیدی رفع اشکال، مدیریت منابع و ارورهای رایج در محیط‌های لینوکس و استقرار کانتینری Docker به همراه پایگاه داده PostgreSQL است.

---

## ۱. خطاهای متداول در اتصال به Docker Daemon

یکی از رایج‌ترین خطاهایی که مهندسان DevOps در هنگام اجرای دستورات با آن مواجه می‌شوند، خطای عدم دسترسی به سوکت داکر است:

```bash
permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/containers/json": dial unix /var/run/docker.sock: connect: permission denied
```

### علت ریشه‌ای و نحوه رفع مشکل
این خطا ناشی از عدم عضویت کاربر فعلی سیستم در گروه `docker` لینوکس است. برای حل مشکل بدون استفاده مداوم از `sudo`، دستورات زیر را اجرا کنید:

```bash
# ایجاد گروه داکر در صورت عدم وجود
sudo groupadd docker

# افزودن کاربر جاری به گروه
sudo usermod -aG docker $USER

# اعمال تغییرات بدون نیاز به خروج کامل از نشست کاربری
newgrp docker
```

---

## ۲. خطاهای شبکه و خطای HTTP 502 Bad Gateway

هنگامی که از Nginx یا Envoy به عنوان Reverse Proxy برای سرویس‌های کانتینری استفاده می‌شود، خطای `502 Bad Gateway` نشان‌دهنده قطع ارتباط بین پروکسی و کانتینر بالادستی (Upstream) است:

```nginx
# نمونه لاگ خطای Nginx
[error] 1402#1402: *1 connect() failed (111: Connection refused) while connecting to upstream, client: 192.168.1.105, server: api.internal, request: "GET /api/v1/health HTTP/1.1", upstream: "http://172.18.0.4:8000/api/v1/health", host: "api.internal"
```

### چک‌لیست بررسی شبکه کانتینرها:
| پارامتر بررسی | دستور عیب‌یابی | حالت نرمال |
|---|---|---|
| وضعیت کانتینر | `docker ps -a --filter status=running` | وضعیت `Up X minutes` |
| شبکه مشترک | `docker network inspect app_network` | مشاهده IP هر دو کانتینر |
| پورت ورودی کانتینر | `netstat -tuln` درون کانتینر | شنود روی `0.0.0.0` نه `127.0.0.1` |

در استقرار با Docker Compose، اطمینان حاصل کنید که هر دو سرویس در یک Bridge Network مشترک تعریف شده‌اند:

```yaml
version: '3.8'

services:
  web-proxy:
    image: nginx:alpine
    ports:
      - "80:80"
    networks:
      - internal_net
    depends_on:
      - backend-api

  backend-api:
    image: python:3.11-slim
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    networks:
      - internal_net

networks:
  internal_net:
    driver: bridge
```

---

## ۳. خطای حافظه اشتراکی و فلگ `--ipc=host`

در بارهای پردازشی با ترافیک بالا یا سرویس‌های هوش مصنوعی (مانند PyTorch و vLLM) و پردازش‌های سنگین، کانتینر با خطای پر شدن `/dev/shm` کرش می‌کند:

```
RuntimeError: DataLoader worker (pid 1842) is killed by signal: Bus error. It is possible that the dataloader's workers ran out of shared memory. Please try increasing the shared memory limit via --shm-size or --ipc=host.
```

### راهکار پیکربندی
داکر به صورت پیش‌فرض حافظه اشتراکی را به 64MB محدود می‌کند. دو راهکار اصلی وجود دارد:
1. استفاده از فلگ اختصاصی برای اشتراک حافظه با سیستم‌عامل میزبان:
   ```bash
   docker run --rm -it --ipc=host my-ai-worker:latest
   ```
2. افزایش صریح سقف سایز حافظه اشتراکی:
   ```bash
   docker run --rm -it --shm-size=8g my-ai-worker:latest
   ```

در Docker Compose:
```yaml
services:
  ai-inference:
    image: vllm/vllm-openai:latest
    shm_size: '8gb'
    ipc: host
    restart: always
```

---

## ۴. خطاهای PostgreSQL: ارور ERR_DB_TIMEOUT و محدودیت اتصالات

هنگامی که استخر اتصالات پایگاه داده (Connection Pool) اشباع می‌شود یا تراکنش‌های طولانی‌مدت وجود دارند، اپلیکیشن با خطاهای زیر مواجه می‌شود:

- `ERR_DB_TIMEOUT: connection pool exhausted after 30000ms`
- `SQLSTATE[08006]: FATAL: remaining connection slots are reserved for non-replication superuser connections`

### تنظیمات حیاتی در `postgresql.conf`
برای برطرف کردن خطای کمبود اسلات اتصال و تنظیم بهینه حافظه، پارامترهای زیر را ویرایش کنید:

```ini
# آدرس‌های شنود شبکه
listen_addresses = '*'

# حداکثر تعداد اتصالات هم‌زمان
max_connections = 250

# تخصیص بافر اشتراکی بهینه (معمولاً ۲۵٪ از کل RAM سرور)
shared_buffers = 4GB

# حافظه برای عملیات سورت و ایندکس داخلی
work_mem = 16MB
maintenance_work_mem = 512MB

# جلوگیری از قفل شدن اتصالات نامعتبر
idle_in_transaction_session_timeout = 60000
statement_timeout = 30000
```

### شناسایی اتصالات معلق و طولانی با کوئری مانیتورینگ:

```sql
SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '10 seconds'
  AND state != 'idle';
```

برای لغو تراکنش‌های گیر کرده:
```sql
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = 14285;
```

---

## ۵. سیاست‌های راه‌اندازی مجدد (Restart Policies) و مدیریت لاگ‌ها

جهت جلوگیری از پر شدن دیسک سرور به دلیل لاگ‌های حجیم کانتینرها، همیشه از لاگ‌درایور با قابلیت چرخش (Log Rotation) استفاده کنید:

```yaml
services:
  app-backend:
    image: my-service:v1.0
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
```
