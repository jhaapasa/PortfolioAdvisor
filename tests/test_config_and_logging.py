from __future__ import annotations

import logging
from pathlib import Path

import pytest

from portfolio_advisor.config import Settings
from portfolio_advisor.logging_config import configure_logging


def test_settings_dir_validation(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    s = Settings(input_dir=str(in_dir), output_dir=str(out_dir))
    s.ensure_directories()
    assert out_dir.exists()

    with pytest.raises(FileNotFoundError):
        Settings(input_dir=str(tmp_path / "nope"), output_dir=str(out_dir)).ensure_directories()


def test_configure_logging_plain_and_json(caplog):
    caplog.set_level(logging.INFO)
    configure_logging(level="INFO", fmt="plain")
    logging.getLogger("portfolio_advisor.test").info("hello")
    assert any("hello" in rec.message for rec in caplog.records)

    caplog.clear()
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("portfolio_advisor.test").info("world")
    assert any("world" in rec.message for rec in caplog.records)
