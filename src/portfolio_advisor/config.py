from __future__ import annotations

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _permissive_json_loads(value: str):  # pragma: no cover - exercised via settings initialization
    """Return raw string for non-JSON-looking values so validators can handle them.

    This prevents pydantic-settings from eagerly JSON-decoding comma-delimited strings
    for complex fields like list[str]. If the value appears to be JSON (starts with
    '{' or '['), attempt normal json parsing; otherwise return the original string.
    """
    if isinstance(value, str):
        s = value.strip()
        if not s or s[0] not in "[{":
            return value
    import json as _json

    try:
        return _json.loads(value)
    except Exception:
        return value


class Settings(BaseSettings):
    """Application settings loaded from environment with sensible defaults.

    CLI-provided overrides should be passed as keyword args to `Settings` which will
    supersede environment/.env values.
    """

    # IO
    input_dir: str
    output_dir: str
    # Optional override; defaults to <output_dir>/portfolio when unset
    portfolio_dir: str | None = Field(default=None, alias="PORTFOLIO_DIR")

    # OpenAI / LLM
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    request_timeout_s: int = Field(default=60, alias="REQUEST_TIMEOUT_S")
    max_tokens: int | None = Field(default=None, alias="MAX_TOKENS")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")

    # Parser configuration
    parser_max_retries: int = Field(default=2, alias="PARSER_MAX_RETRIES")
    parser_max_doc_chars: int = Field(default=20000, alias="PARSER_MAX_DOC_CHARS")

    # Resolver / Polygon configuration
    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")
    polygon_base_url: str | None = Field(default=None, alias="POLYGON_BASE_URL")
    polygon_timeout_s: int = Field(default=10, alias="POLYGON_TIMEOUT_S")
    resolver_default_locale: str = Field(default="us", alias="RESOLVER_DEFAULT_LOCALE")
    # Comma-separated string to avoid JSON array parsing in .env
    resolver_preferred_mics: str = Field(default="XNAS,XNYS,ARCX", alias="RESOLVER_PREFERRED_MICS")
    resolver_confidence_threshold: float = Field(default=0.6, alias="RESOLVER_CONFIDENCE_THRESHOLD")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="plain", alias="LOG_FORMAT")
    verbose: bool = Field(default=False, alias="VERBOSE")
    agent_progress: bool = Field(default=False, alias="AGENT_PROGRESS")
    log_libraries: bool = Field(default=False, alias="LOG_LIBRARIES")
    # Caching
    skip_llm_cache: bool = Field(default=False, alias="SKIP_LLM_CACHE")

    # Optional analysis feature flags
    wavelet: bool = Field(default=False, alias="WAVELET")
    wavelet_level: int = Field(default=5, alias="WAVELET_LEVEL")
    fetch_news: bool = Field(default=True, alias="FETCH_NEWS")

    # Ollama configuration for article text extraction
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_timeout_s: int = Field(default=120, alias="OLLAMA_TIMEOUT_S")
    extraction_model: str = Field(default="milkey/reader-lm-v2:Q8_0", alias="EXTRACTION_MODEL")
    extraction_batch_size: int = Field(default=10, alias="EXTRACTION_BATCH_SIZE")
    extraction_max_workers: int = Field(default=4, alias="EXTRACTION_MAX_WORKERS")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        json_loads=_permissive_json_loads,
        populate_by_name=True,
    )

    @field_validator("input_dir", "output_dir")
    @classmethod
    def _validate_dirs(cls, value: str) -> str:
        if not value:
            msg = "Input and output directories must be provided."
            raise ValueError(msg)
        return value

    # No-op: parsing is handled in resolver builder to keep .env simple

    def ensure_directories(self) -> None:
        if not os.path.isdir(self.input_dir):
            msg = f"Input directory does not exist or is not a directory: {self.input_dir}"
            raise FileNotFoundError(msg)
        os.makedirs(self.output_dir, exist_ok=True)
        # Default portfolio_dir if not provided
        if not self.portfolio_dir:
            self.portfolio_dir = os.path.join(self.output_dir, "portfolio")
        os.makedirs(self.portfolio_dir, exist_ok=True)

    @field_validator("wavelet_level")
    @classmethod
    def _validate_wavelet_level(cls, value: int) -> int:
        # Allow at least 1..8; can be extended later
        if int(value) < 1 or int(value) > 8:
            raise ValueError("WAVELET_LEVEL must be between 1 and 8")
        return int(value)
