from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from portfolio_advisor.cli import main


def test_cli_success(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rc = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "DEBUG",
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    output_path = Path(out.out.strip())
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "Portfolio Analysis Report" in content


def _write_minimal_ticker_scaffold(
    base: Path, slug: str, meta: dict, artifacts: Iterable[str], *, include_report: bool = True
) -> Path:
    ticker_dir = base / slug
    (ticker_dir / "analysis").mkdir(parents=True, exist_ok=True)
    if include_report:
        (ticker_dir / "report").mkdir(parents=True, exist_ok=True)
        (ticker_dir / "report" / "candle_ohlcv_1y.png").write_bytes(b"\x89PNGfake")
    (ticker_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for artifact in artifacts:
        (ticker_dir / "analysis" / artifact).write_text(json.dumps({}), encoding="utf-8")
    return ticker_dir


def test_cli_stock_mode_wavelet(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, list[str] | None] = {"requested": None}

    def _fake_update_instrument(settings, instrument, requested_artifacts=None):  # noqa: ANN001
        captured["requested"] = requested_artifacts
        slug = "cid-stocks-us-composite-aapl"
        meta = {
            "instrument_id": instrument.get("instrument_id"),
            "primary_ticker": instrument.get("primary_ticker"),
            "artifacts": {"primary.ohlc_daily": {"last_updated": "2025-01-02"}},
        }
        _write_minimal_ticker_scaffold(
            base=Path(settings.output_dir) / "stocks" / "tickers",
            slug=slug,
            meta=meta,
            artifacts=(
                "returns.json",
                "wavelet_coeffs.json",
                "wavelet_variance_histogram.json",
            ),
            include_report=True,
        )

    monkeypatch.setattr("portfolio_advisor.cli.update_instrument", _fake_update_instrument)

    rc = main(
        [
            "--mode",
            "stock",
            "--ticker",
            "AAPL",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--wavelet",
            "--wavelet-level",
            "4",
            "--log-level",
            "DEBUG",
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    ticker_dir = Path(out.out.strip())
    assert ticker_dir.exists()
    assert captured["requested"] == [
        "primary.ohlc_daily",
        "analysis.returns",
        "analysis.volatility",
        "analysis.sma_20_50_100_200",
        "analysis.wavelet_modwt_j5_sym4",
    ]


def test_cli_stock_mode_requires_identifier(tmp_path: Path, capsys):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    rc = main(
        [
            "--mode",
            "stock",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--ticker or --instrument-id is required" in err


def test_cli_stock_mode_instrument_id_parsing(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nMSFT,20\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, dict | None] = {"instrument": None}

    def _fake_update(settings, instrument, requested_artifacts=None):  # noqa: ANN001
        captured["instrument"] = instrument
        base = Path(settings.output_dir) / "stocks" / "tickers"
        _write_minimal_ticker_scaffold(
            base=base,
            slug="cid-stocks-us-composite-msft",
            meta={
                "instrument_id": instrument.get("instrument_id"),
                "primary_ticker": instrument.get("primary_ticker"),
            },
            artifacts=("returns.json",),
            include_report=False,
        )

    monkeypatch.setattr("portfolio_advisor.cli.update_instrument", _fake_update)

    rc = main(
        [
            "--mode",
            "stock",
            "--instrument-id",
            "cid:stocks:us:composite:MSFT",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    assert Path(out.out.strip()).exists()
    assert captured["instrument"] == {
        "instrument_id": "cid:stocks:us:composite:MSFT",
        "primary_ticker": "MSFT",
    }


def test_cli_stock_mode_detects_existing_meta(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    base = output_dir / "stocks" / "tickers"
    meta = {
        "instrument_id": "cid:stocks:us:composite:AAPL",
        "primary_ticker": "AAPL",
        "artifacts": {"primary.ohlc_daily": {"last_updated": "2025-01-02"}},
    }
    _write_minimal_ticker_scaffold(
        base=base,
        slug="cid-stocks-us-composite-aapl",
        meta=meta,
        artifacts=("returns.json",),
        include_report=False,
    )

    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,5\n", encoding="utf-8")

    captured: dict[str, dict | None] = {"instrument": None}

    def _fake_update(settings, instrument, requested_artifacts=None):  # noqa: ANN001
        captured["instrument"] = instrument

    monkeypatch.setattr("portfolio_advisor.cli.update_instrument", _fake_update)

    rc = main(
        [
            "--mode",
            "stock",
            "--ticker",
            "AAPL",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr()
    assert Path(out.out.strip()).exists()
    assert captured["instrument"] == {
        "instrument_id": "cid:stocks:us:composite:AAPL",
        "primary_ticker": "AAPL",
    }


def test_cli_missing_input_dir(tmp_path: Path, capsys):
    input_dir = tmp_path / "missing"
    output_dir = tmp_path / "out"
    rc = main(["--input-dir", str(input_dir), "--output-dir", str(output_dir)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error:" in err


def test_cli_stock_mode_fetch_news_flag(tmp_path: Path, capsys, monkeypatch):
    """Test that --fetch-news and --no-fetch-news flags work correctly."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    captured_settings: dict[str, object | None] = {"settings": None}

    def _fake_update(settings, instrument, requested_artifacts=None):  # noqa: ANN001
        captured_settings["settings"] = settings

    monkeypatch.setattr("portfolio_advisor.cli.update_instrument", _fake_update)
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")

    base_args = [
        "--mode",
        "stock",
        "--ticker",
        "AAPL",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
    ]

    def _run(extra: list[str], expected: bool | None) -> None:
        captured_settings["settings"] = None
        rc = main(base_args + extra)
        assert rc == 0
        assert captured_settings["settings"] is not None
        assert captured_settings["settings"].fetch_news is expected

    _run(["--fetch-news"], True)
    _run(["--no-fetch-news"], False)
    _run([], True)


def test_cli_extract_text_requires_identifier(tmp_path: Path, capsys):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    rc = main(
        [
            "--mode",
            "extract-text",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "--ticker, --instrument-id, or --all-stocks" in err


def test_cli_extract_text_single_ticker(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    # Pre-create ticker meta to exercise slug discovery path
    slug = "cid-stocks-us-composite-aapl"
    ticker_dir = output_dir / "stocks" / "tickers" / slug
    ticker_dir.mkdir(parents=True, exist_ok=True)
    (ticker_dir / "meta.json").write_text(
        json.dumps(
            {
                "instrument_id": "cid:stocks:us:composite:AAPL",
                "primary_ticker": "AAPL",
            }
        ),
        encoding="utf-8",
    )

    class DummyOllama:
        def __enter__(self):
            return self

        def __exit__(self, *exc):  # noqa: D401 - simple stub cleanup
            return False

    class DummyExtractor:
        instances: list[DummyExtractor] = []

        def __init__(self, *args, **kwargs):
            self.calls: list[tuple[str, tuple, dict]] = []
            DummyExtractor.instances.append(self)

        def extract_all_articles(self, ticker_slug, force=False, batch_size=0):
            self.calls.append(
                ("single", (ticker_slug,), {"force": force, "batch_size": batch_size})
            )
            return {
                "total_articles": 2,
                "extracted": 2,
                "skipped": 0,
                "errors": 0,
            }

        def extract_portfolio_articles(self, *args, **kwargs):
            raise AssertionError("unexpected portfolio extraction call")

    monkeypatch.setattr(
        "portfolio_advisor.services.ollama_service.OllamaService",
        lambda *a, **k: DummyOllama(),
    )
    monkeypatch.setattr(
        "portfolio_advisor.stocks.article_extraction.ArticleTextExtractionService",
        DummyExtractor,
    )

    rc = main(
        [
            "--mode",
            "extract-text",
            "--ticker",
            "AAPL",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--force-extraction",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Processed" in out
    extractor = DummyExtractor.instances[-1]
    assert extractor.calls
    call = extractor.calls[0]
    assert call[0] == "single"
    assert call[1][0] == slug
    assert call[2]["force"] is True


def test_cli_extract_text_all_stocks(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()

    class DummyOllama:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class DummyExtractor:
        instances: list[DummyExtractor] = []

        def __init__(self, *args, **kwargs):
            self.calls: list[tuple[str, tuple, dict]] = []
            DummyExtractor.instances.append(self)

        def extract_portfolio_articles(self, portfolio_path, force=False, max_workers=0):
            self.calls.append(
                ("portfolio", (portfolio_path,), {"force": force, "max_workers": max_workers})
            )
            return {
                "total_tickers": 1,
                "processed_tickers": 1,
                "total_articles": 3,
                "extracted": 3,
                "skipped": 0,
                "errors": 0,
            }

        def extract_all_articles(self, *args, **kwargs):
            raise AssertionError("unexpected single extraction call")

    monkeypatch.setattr(
        "portfolio_advisor.services.ollama_service.OllamaService",
        lambda *a, **k: DummyOllama(),
    )
    monkeypatch.setattr(
        "portfolio_advisor.stocks.article_extraction.ArticleTextExtractionService",
        DummyExtractor,
    )

    rc = main(
        [
            "--mode",
            "extract-text",
            "--all-stocks",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Processed 1 tickers" in out
    extractor = DummyExtractor.instances[-1]
    call = extractor.calls[0]
    assert call[0] == "portfolio"
    assert call[2]["force"] is False


def test_cli_wavelet_level_implies_wavelet(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "positions.csv").write_text("symbol,shares\nAAPL,10\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict[str, list[str] | None] = {"requested": None}

    def _fake_update(settings, instrument, requested_artifacts=None):  # noqa: ANN001
        captured["requested"] = requested_artifacts
        base = Path(settings.output_dir) / "stocks" / "tickers"
        _write_minimal_ticker_scaffold(
            base=base,
            slug="cid-stocks-us-composite-aapl",
            meta={
                "instrument_id": instrument.get("instrument_id"),
                "primary_ticker": instrument.get("primary_ticker"),
            },
            artifacts=("returns.json",),
            include_report=False,
        )

    monkeypatch.setattr("portfolio_advisor.cli.update_instrument", _fake_update)

    rc = main(
        [
            "--mode",
            "stock",
            "--ticker",
            "AAPL",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--wavelet-level",
            "3",
        ]
    )

    assert rc == 0
    assert captured["requested"] == [
        "primary.ohlc_daily",
        "analysis.returns",
        "analysis.volatility",
        "analysis.sma_20_50_100_200",
        "analysis.wavelet_modwt_j5_sym4",
    ]


def test_cli_portfolio_mode_failure(tmp_path: Path, capsys, monkeypatch):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    def _boom(**kwargs):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr("portfolio_advisor.cli.analyze_portfolio", _boom)

    rc = main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == 1
    captured = capsys.readouterr()
    assert "Error: boom" in captured.err
