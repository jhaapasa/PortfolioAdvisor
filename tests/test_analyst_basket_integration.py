from __future__ import annotations

from portfolio_advisor.agents.analyst import analyst_node


def test_analyst_includes_basket_highlights(monkeypatch):
    class DummyLLM:
        def invoke(self, prompt: str):
            class R:
                def __init__(self, content: str):
                    self.content = content

            return R(prompt)

    def fake_get_llm(_settings):
        return DummyLLM()

    monkeypatch.setattr("portfolio_advisor.agents.analyst.get_llm", fake_get_llm)

    state = {
        "settings": object(),
        "raw_docs": [],
        "resolved_holdings": [],
        "basket_reports": [
            {
                "id": "growth_tech",
                "label": "Growth Tech",
                "slug": "growth-tech",
                "averages": {"d1": 0.01, "d5": 0.02},
                "summary_text": "- AAPL led gains.\n- NVDA dipped slightly.",
            }
        ],
    }
    out = analyst_node(state)
    content = out.get("analysis", "")
    assert "Basket Highlights" in content
    assert "Growth Tech" in content


