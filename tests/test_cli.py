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

    # Test with --fetch-news (should be True)
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
            "--fetch-news",
        ]
    )
    assert rc == 0
    assert captured_settings["settings"] is not None
    assert captured_settings["settings"].fetch_news is True

    # Test with --no-fetch-news (should be False)
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
            "--no-fetch-news",
        ]
    )
    assert rc == 0
    assert captured_settings["settings"] is not None
    assert captured_settings["settings"].fetch_news is False

    # Test default (should be True based on Settings default)
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
    assert captured_settings["settings"] is not None
    assert captured_settings["settings"].fetch_news is True
