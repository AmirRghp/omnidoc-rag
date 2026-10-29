"""Application configuration loader using Pydantic Settings and YAML fallback."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def load_yaml_config(yaml_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file if it exists."""
    if not yaml_path.is_file():
        return {}
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """Application settings with environment variable precedence over YAML defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Server & Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    api_key: str = Field(default="dev_insecure_local_key_replace_in_prod", alias="API_KEY")
    admin_api_key: str | None = Field(default=None, alias="ADMIN_API_KEY")
    api_key_sha256: str | None = Field(default=None, alias="API_KEY_SHA256")
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    upload_rate_limit_per_hour: int = Field(default=5, alias="UPLOAD_RATE_LIMIT_PER_HOUR")

    # Qdrant Vector Store
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_https: bool | None = Field(default=None, alias="QDRANT_HTTPS")
    collection_name: str = Field(default="tech_docs_bilingual", alias="COLLECTION_NAME")

    # Redis Cache
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str | None = Field(default=None, alias="REDIS_PASSWORD")
    cache_ttl_seconds: int = Field(default=86400, alias="CACHE_TTL_SECONDS")
    semantic_cache_enabled: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")

    # LLM Inference (vLLM OpenAI-compatible endpoint)
    llm_base_url: str = Field(default="http://localhost:8000/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="dummy-local-key", alias="LLM_API_KEY")
    llm_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", alias="LLM_MODEL")

    # Observability (Langfuse)
    langfuse_public_key: str | None = Field(default="pk-lf-local", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default="sk-lf-local", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")

    # Retrieval Hyperparameters
    top_k_dense: int = Field(default=20, alias="TOP_K_DENSE")
    top_k_sparse: int = Field(default=20, alias="TOP_K_SPARSE")
    rrf_k: int = Field(default=60, alias="RRF_K")
    rerank_top_n: int = Field(default=5, alias="RERANK_TOP_N")
    dense_model_name: str = Field(default="BAAI/bge-m3", alias="DENSE_MODEL_NAME")
    rerank_model_name: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANK_MODEL_NAME")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # CRAG Evaluation Thresholds
    crag_correct_threshold: float = Field(default=0.70, alias="CRAG_CORRECT_THRESHOLD")
    crag_ambiguous_threshold: float = Field(default=0.35, alias="CRAG_AMBIGUOUS_THRESHOLD")
    max_query_transformations: int = Field(default=1, alias="MAX_QUERY_TRANSFORMATIONS")

    @field_validator(
        "qdrant_api_key",
        "redis_password",
        "langfuse_public_key",
        "langfuse_secret_key",
        "qdrant_https",
        "admin_api_key",
        "api_key_sha256",
        mode="before",
    )
    @classmethod
    def empty_str_to_none(cls, value: Any) -> Any:
        """Convert empty strings to None."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("crag_correct_threshold", "crag_ambiguous_threshold")
    @classmethod
    def validate_thresholds(cls, value: float) -> float:
        """Ensure thresholds are between 0.0 and 1.0."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {value}")
        return value

    @field_validator(
        "top_k_dense",
        "top_k_sparse",
        "rrf_k",
        "rerank_top_n",
        "rate_limit_per_minute",
        "upload_rate_limit_per_hour",
    )
    @classmethod
    def validate_positive_integers(cls, value: int) -> int:
        """Ensure retrieval and rate-limit hyperparameter counts are strictly positive."""
        if value <= 0:
            raise ValueError(f"Value must be positive, got {value}")
        return value


def create_settings(
    config_yaml_path: Path | None = None,
    **kwargs: Any,
) -> Settings:
    """Instantiate Settings merging YAML configuration and environment variables.

    Precedence order: explicit kwargs > environment variables > YAML config > field defaults.
    """
    yaml_path = config_yaml_path or Path("configs/config.yaml")
    yaml_data = load_yaml_config(yaml_path)

    # Flatten nested YAML structure into flat settings dictionary
    flattened_yaml: dict[str, Any] = {}
    if "app" in yaml_data and isinstance(yaml_data["app"], dict):
        if "environment" in yaml_data["app"]:
            flattened_yaml["environment"] = yaml_data["app"]["environment"]
        if "host" in yaml_data["app"]:
            flattened_yaml["app_host"] = yaml_data["app"]["host"]
        if "port" in yaml_data["app"]:
            flattened_yaml["app_port"] = yaml_data["app"]["port"]

    if "retrieval" in yaml_data and isinstance(yaml_data["retrieval"], dict):
        retrieval = yaml_data["retrieval"]
        for key in (
            "collection_name",
            "top_k_dense",
            "top_k_sparse",
            "rrf_k",
            "rerank_top_n",
            "dense_model_name",
            "rerank_model_name",
            "embedding_dimension",
        ):
            if key in retrieval:
                flattened_yaml[key] = retrieval[key]

    if "crag" in yaml_data and isinstance(yaml_data["crag"], dict):
        crag = yaml_data["crag"]
        if "correct_threshold" in crag:
            flattened_yaml["crag_correct_threshold"] = crag["correct_threshold"]
        if "ambiguous_threshold" in crag:
            flattened_yaml["crag_ambiguous_threshold"] = crag["ambiguous_threshold"]
        if "max_query_transformations" in crag:
            flattened_yaml["max_query_transformations"] = crag["max_query_transformations"]

    if "cache" in yaml_data and isinstance(yaml_data["cache"], dict):
        cache = yaml_data["cache"]
        if "ttl_seconds" in cache:
            flattened_yaml["cache_ttl_seconds"] = cache["ttl_seconds"]
        if "semantic_cache_enabled" in cache:
            flattened_yaml["semantic_cache_enabled"] = cache["semantic_cache_enabled"]

    # Filter out YAML keys that are already overridden in os.environ so env vars win over YAML
    yaml_without_env_overrides: dict[str, Any] = {}
    for field_name, val in flattened_yaml.items():
        field_info = Settings.model_fields.get(field_name)
        env_alias = (field_info.alias if field_info and field_info.alias else field_name).upper()
        if env_alias not in os.environ and field_name.upper() not in os.environ:
            yaml_without_env_overrides[field_name] = val

    merged = {**yaml_without_env_overrides, **kwargs}
    return Settings(**merged)


@lru_cache
def get_settings() -> Settings:
    """Cached global singleton instance of application settings."""
    return create_settings()


settings = get_settings()
