"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config import Settings, create_settings


def test_default_settings() -> None:
    """Verify default architectural configurations match specifications."""
    settings = Settings()
    assert settings.top_k_dense == 20
    assert settings.top_k_sparse == 20
    assert settings.rrf_k == 60
    assert settings.rerank_top_n == 5
    assert settings.crag_correct_threshold == 0.70
    assert settings.crag_ambiguous_threshold == 0.35
    assert settings.max_query_transformations == 1
    assert settings.embedding_dimension == 1024
    assert settings.dense_model_name == "BAAI/bge-m3"
    assert settings.rerank_model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.collection_name == "tech_docs_bilingual"


def test_create_settings_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that YAML configuration file properly populates settings."""
    for env_key in ("ENVIRONMENT", "APP_HOST", "APP_PORT"):
        monkeypatch.delenv(env_key, raising=False)
    custom_yaml = tmp_path / "test_config.yaml"
    custom_yaml.write_text(
        """
app:
  environment: "staging"
  host: "127.0.0.1"
  port: 9000

retrieval:
  top_k_dense: 35
  top_k_sparse: 25
  rrf_k: 50
  rerank_top_n: 10
  embedding_dimension: 1024

crag:
  correct_threshold: 0.85
  ambiguous_threshold: 0.40
  max_query_transformations: 3

cache:
  ttl_seconds: 3600
  semantic_cache_enabled: false
""",
        encoding="utf-8",
    )

    settings = create_settings(config_yaml_path=custom_yaml)
    assert settings.environment == "staging"
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 9000
    assert settings.top_k_dense == 35
    assert settings.top_k_sparse == 25
    assert settings.rrf_k == 50
    assert settings.rerank_top_n == 10
    assert settings.crag_correct_threshold == 0.85
    assert settings.crag_ambiguous_threshold == 0.40
    assert settings.max_query_transformations == 3
    assert settings.cache_ttl_seconds == 3600
    assert settings.semantic_cache_enabled is False


def test_env_var_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify environment variables take precedence over both field defaults and YAML."""
    custom_yaml = tmp_path / "test_config.yaml"
    custom_yaml.write_text(
        "retrieval:\n  top_k_dense: 15\napp:\n  port: 8000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TOP_K_DENSE", "42")
    monkeypatch.setenv("APP_PORT", "9090")
    monkeypatch.setenv("LLM_BASE_URL", "http://remote-vllm:8000/v1")
    monkeypatch.setenv("API_KEY", "super-secret-token")

    settings = create_settings(config_yaml_path=custom_yaml)
    assert settings.top_k_dense == 42
    assert settings.app_port == 9090
    assert settings.llm_base_url == "http://remote-vllm:8000/v1"
    assert settings.api_key == "super-secret-token"


def test_invalid_crag_threshold_validation() -> None:
    """Verify thresholds outside 0.0 to 1.0 raise validation errors."""
    with pytest.raises(ValidationError):
        Settings(CRAG_CORRECT_THRESHOLD=1.5)

    with pytest.raises(ValidationError):
        Settings(CRAG_AMBIGUOUS_THRESHOLD=-0.1)


def test_invalid_positive_integer_validation() -> None:
    """Verify non-positive integers raise validation errors."""
    with pytest.raises(ValidationError):
        Settings(TOP_K_DENSE=0)

    with pytest.raises(ValidationError):
        Settings(RRF_K=-5)
