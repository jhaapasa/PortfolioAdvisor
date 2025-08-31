from __future__ import annotations

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment with sensible defaults.

    CLI-provided overrides should be passed as keyword args to `Settings` which will
    supersede environment/.env values.
    """

    # IO
    input_dir: str
    output_dir: str

    # Gemini / LLM
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_api_base: str | None = Field(default=None, alias="GEMINI_API_BASE")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    request_timeout_s: int = Field(default=60, alias="REQUEST_TIMEOUT_S")
    max_tokens: int | None = Field(default=None, alias="MAX_TOKENS")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")

    # Parser configuration
    parser_max_retries: int = Field(default=2, alias="PARSER_MAX_RETRIES")
    parser_max_doc_chars: int = Field(default=20000, alias="PARSER_MAX_DOC_CHARS")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="plain", alias="LOG_FORMAT")

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    @field_validator("input_dir", "output_dir")
    @classmethod
    def _validate_dirs(cls, value: str) -> str:
        if not value:
            msg = "Input and output directories must be provided."
            raise ValueError(msg)
        return value

    def ensure_directories(self) -> None:
        if not os.path.isdir(self.input_dir):
            msg = f"Input directory does not exist or is not a directory: {self.input_dir}"
            raise FileNotFoundError(msg)
        os.makedirs(self.output_dir, exist_ok=True)
