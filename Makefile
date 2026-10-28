.PHONY: help up down logs lint format test test-unit test-integration eval clean

help:
	@echo "OmniDoc RAG Development Commands:"
	@echo "  make up               - Start infrastructure services (Qdrant, Redis, Postgres, Langfuse)"
	@echo "  make down             - Stop infrastructure services"
	@echo "  make logs             - Tail infrastructure container logs"
	@echo "  make lint             - Run ruff check and mypy type verification"
	@echo "  make format           - Format code using ruff"
	@echo "  make test             - Run complete pytest test suite"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make eval             - Run Ragas evaluation benchmarks"
	@echo "  make clean            - Remove cache and temporary artifacts"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

lint:
	ruff check .
	ruff format --check .
	mypy src

format:
	ruff format .
	ruff check --fix .

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

eval:
	python evals/run_benchmark.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
