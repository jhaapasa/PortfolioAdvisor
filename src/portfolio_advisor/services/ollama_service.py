"""Service for interacting with Ollama API."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for interacting with the Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 60):
        """Initialize ollama service with configurable endpoint.

        Args:
            base_url: Base URL for ollama API (default: http://localhost:11434)
            timeout_s: Request timeout in seconds (default: 60)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure client is closed."""
        self._client.close()

    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """Send a generation request to ollama.

        Args:
            model: Model name (e.g., "milkey/reader-lm-v2:Q8_0")
            prompt: The prompt text
            system: Optional system prompt
            temperature: Generation temperature (default: 0.1 for consistency)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to ollama

        Returns:
            Generated text response

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response format is unexpected
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if system:
            payload["system"] = system

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        # Add any additional options
        if kwargs:
            payload.update(kwargs)

        logger.debug(f"Sending generation request to {url} with model {model}")

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            if "response" not in data:
                raise ValueError(f"Unexpected response format: {data}")

            return data["response"]

        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling ollama: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling ollama: {e}")
            raise

    def is_available(self) -> bool:
        """Check if ollama service is available.

        Returns:
            True if service is reachable, False otherwise
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names

        Raises:
            httpx.HTTPError: If the request fails
        """
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise

    def model_exists(self, model: str) -> bool:
        """Check if a specific model exists.

        Args:
            model: Model name to check

        Returns:
            True if model exists, False otherwise
        """
        try:
            models = self.list_models()
            return model in models
        except Exception:
            return False
