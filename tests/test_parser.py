from __future__ import annotations

import json
from typing import Any

from portfolio_advisor.agents.parser import parse_one_node


class DummyLLM:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def invoke(self, prompt: str):  # simple sync path used by parser
        class R:
            def __init__(self, content: str):
                self.content = content

        return R(json.dumps(self.payload))


def test_parser_basic_concat(monkeypatch):
    # Two simple docs producing one holding each
    payloads = [
        {
            "source_doc_id": "doc1.txt",
            "holdings": [
                {
                    "name": "Apple Inc.",
                    "confidence": 0.9,
                    "source_doc_id": "doc1.txt",
                    "primary_ticker": "AAPL",
                    "basket": "[none]",
                }
            ],
            "errors": [],
        },
        {
            "source_doc_id": "doc2.csv",
            "holdings": [
                {
                    "name": "Microsoft Corp",
                    "confidence": 0.85,
                    "source_doc_id": "doc2.csv",
                    "primary_ticker": "MSFT",
                    "basket": "Tech",
                }
            ],
            "errors": [],
        },
    ]

    # Patch get_llm to return deterministic dummy responses, rotating per call
    calls = {"i": 0}

    def fake_get_llm(_settings):
        idx = calls["i"]
        calls["i"] += 1
        return DummyLLM(payloads[idx])

    monkeypatch.setattr("portfolio_advisor.agents.parser.get_llm", fake_get_llm)

    class S:
        parser_max_concurrency = 2
        parser_max_rpm = 1000
        parser_max_retries = 0
        parser_max_doc_chars = 5000

    settings = S()
    out1 = parse_one_node(
        {"settings": settings, "doc": {"name": "doc1.txt", "as_text": "Position: Apple"}}
    )
    out2 = parse_one_node(
        {"settings": settings, "doc": {"name": "doc2.csv", "as_text": "MSFT, 100"}}
    )
    holdings = (out1.get("parsed_holdings", []) or []) + (out2.get("parsed_holdings", []) or [])
    assert len(holdings) == 2
    names = {h["name"] for h in holdings}
    assert names == {"Apple Inc.", "Microsoft Corp"}
    baskets = {h["basket"] for h in holdings}
    assert "[none]" in baskets


def test_parser_retry_fix(monkeypatch):
    # First response invalid; second response valid
    bad = "{"  # invalid JSON
    good = json.dumps(
        {
            "source_doc_id": "d.txt",
            "holdings": [
                {"name": "Tesla", "confidence": 0.7, "source_doc_id": "d.txt", "basket": "[none]"}
            ],
            "errors": [],
        }
    )

    class DummyLLM2:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt: str):
            self.calls += 1

            class R:
                def __init__(self, content: str):
                    self.content = content

            return R(bad if self.calls == 1 else good)

    def fake_get_llm(_settings):
        return DummyLLM2()

    monkeypatch.setattr("portfolio_advisor.agents.parser.get_llm", fake_get_llm)

    class S:
        parser_max_concurrency = 1
        parser_max_rpm = 1000
        parser_max_retries = 1
        parser_max_doc_chars = 5000

    state = {"settings": S(), "doc": {"name": "d.txt", "as_text": "TSLA"}}
    out = parse_one_node(state)
    holdings = out.get("parsed_holdings", [])
    assert len(holdings) == 1
    assert holdings[0]["name"] == "Tesla"
