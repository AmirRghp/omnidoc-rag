# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Create non-root user (uid=10001 per docs/SECURITY.md §5.1)
RUN groupadd --gid 10001 omnidoc \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin omnidoc

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app
RUN chown -R omnidoc:omnidoc /app

USER 10001:10001

EXPOSE 8080 3000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
