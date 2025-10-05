from __future__ import annotations

import httpx
import pytest

from portfolio_advisor.services.ollama_service import OllamaService


class DummyResponse:
    def __init__(self, json_data: dict | None = None, status_code: int = 200) -> None:
        self._json = json_data or {}
        self.status_code = status_code

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPError(f"status={self.status_code}")


class DummyClient:
    def __init__(
        self,
        *,
        post_responses: list[DummyResponse | Exception] | None = None,
        get_responses: list[DummyResponse | Exception] | None = None,
    ) -> None:
        self._post_responses = list(post_responses or [])
        self._get_responses = list(get_responses or [])
        self.post_calls: list[tuple[str, dict]] = []
        self.get_calls: list[tuple[str, float | None]] = []
        self.closed = False

    def post(self, url: str, json: dict):
        self.post_calls.append((url, json))
        if not self._post_responses:
            raise RuntimeError("no post response configured")
        response = self._post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url: str, timeout: float | None = None):
        self.get_calls.append((url, timeout))
        if not self._get_responses:
            raise RuntimeError("no get response configured")
        response = self._get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def client_factory(monkeypatch):
    def _factory(*, post_responses=None, get_responses=None) -> DummyClient:
        client = DummyClient(post_responses=post_responses, get_responses=get_responses)

        def _build_client(*_args, **_kwargs):
            return client

        monkeypatch.setattr("portfolio_advisor.services.ollama_service.httpx.Client", _build_client)
        return client

    return _factory


def test_generate_builds_payload_and_returns_text(client_factory):
    client = client_factory(post_responses=[DummyResponse({"response": "hello"})])

    service = OllamaService(base_url="http://ollama.test", timeout_s=30)
    result = service.generate(
        model="reader",
        prompt="Prompt text",
        system="System prompt",
        temperature=0.3,
        max_tokens=128,
        extra_param="value",
    )

    assert result == "hello"
    assert client.post_calls, "expected generate to issue POST"
    url, payload = client.post_calls[0]
    assert url == "http://ollama.test/api/generate"
    assert payload["model"] == "reader"
    assert payload["prompt"] == "Prompt text"
    assert payload["system"] == "System prompt"
    assert payload["options"]["temperature"] == 0.3
    assert payload["options"]["num_predict"] == 128
    assert payload["extra_param"] == "value"


def test_generate_raises_on_unexpected_payload(client_factory):
    client_factory(post_responses=[DummyResponse({"unexpected": "payload"})])
    service = OllamaService(base_url="http://ollama.test")

    with pytest.raises(ValueError):
        service.generate(model="m", prompt="p")


def test_is_available_uses_health_check(client_factory):
    client_factory(get_responses=[DummyResponse(status_code=200)])
    service = OllamaService(base_url="http://ollama.test")
    assert service.is_available() is True

    client_factory(get_responses=[DummyResponse(status_code=503)])
    service = OllamaService(base_url="http://ollama.test")
    assert service.is_available() is False


def test_list_models_and_model_exists(client_factory):
    client = client_factory(
        get_responses=[
            DummyResponse({"models": [{"name": "reader"}, {"name": "other"}]}),
            DummyResponse({"models": [{"name": "reader"}, {"name": "other"}]}),
        ]
    )
    service = OllamaService(base_url="http://ollama.test")

    models = service.list_models()
    assert models == ["reader", "other"]
    assert service.model_exists("reader") is True
    assert service.model_exists("missing") is False
    assert client.get_calls[0][0] == "http://ollama.test/api/tags"
    assert client.get_calls[0][1] is None


def test_context_manager_closes_client(client_factory):
    client = client_factory()
    service = OllamaService(base_url="http://ollama.test")

    with service as svc:
        assert svc is service

    assert client.closed is True
