from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.analyze import analyze_portfolio


def test_end_to_end_summary(tmp_path: Path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()

    (in_dir / "positions.csv").write_text("AAPL, 10\nMSFT, 5", encoding="utf-8")

    payload = {
        "source_doc_id": "positions.csv",
        "holdings": [
            {
                "name": "Apple Inc.",
                "confidence": 0.9,
                "source_doc_id": "positions.csv",
                "primary_ticker": "AAPL",
                "basket": "[none]",
            },
            {
                "name": "Microsoft Corp",
                "confidence": 0.85,
                "source_doc_id": "positions.csv",
                "primary_ticker": "MSFT",
                "basket": "Tech",
            },
        ],
        "errors": [],
    }

    class DummyLLM:
        def invoke(self, prompt: str):

            class R:
                def __init__(self, content: str):
                    self.content = content

            # Parser stage expects JSON; analyst expects free text.
            # Detect via presence of the JSON Schema marker.
            if "JSON Schema for output" in prompt:
                return R(json.dumps(payload))
            return R("Summary of 2 holdings across accounts and baskets.")

    def fake_get_llm(_settings):
        return DummyLLM()

    monkeypatch.setattr("portfolio_advisor.agents.parser.get_llm", fake_get_llm)
    monkeypatch.setattr("portfolio_advisor.agents.analyst.get_llm", fake_get_llm)
    # Stub resolver to return canonical holdings deterministically (no network/env deps)
    from portfolio_advisor.models.canonical import CanonicalHolding

    class _StubResolver:
        def resolve_one(self, parsed: dict):  # noqa: ANN001
            ticker = str(parsed.get("primary_ticker") or "AAPL")
            iid = f"cid:stocks:us:composite:{ticker}"
            return CanonicalHolding(
                instrument_id=iid,
                asset_class="stocks",
                locale="us",
                mic="composite",
                primary_ticker=ticker,
                symbol=ticker,
                polygon_ticker=ticker,
                company_name=str(parsed.get("name") or ticker),
                currency=str(parsed.get("currency") or "USD"),
                as_of=parsed.get("as_of"),
                quantity=parsed.get("quantity"),
                weight=parsed.get("weight"),
                account=parsed.get("account"),
                basket=parsed.get("basket"),
                source_doc_id=parsed.get("source_doc_id"),
                resolution_confidence=1.0,
                resolution_notes="test_stub",
                identifiers=None,
            )

    def _fake_build_resolver(_settings):  # noqa: ANN001
        return _StubResolver()

    monkeypatch.setattr("portfolio_advisor.agents.resolver._build_resolver", _fake_build_resolver)
    # Avoid hitting stocks update pipeline/network in this end-to-end test
    monkeypatch.setattr(
        "portfolio_advisor.graph.update_all_for_instruments", lambda *_args, **_kwargs: None
    )

    analyze_portfolio(input_dir=str(in_dir), output_dir=str(out_dir))
    content = (out_dir / "analysis.md").read_text(encoding="utf-8")
    assert "Summary of 2 holdings" in content
