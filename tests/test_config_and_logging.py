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


def test_settings_empty_dir_validation():
    """Test that empty input/output directories raise validation error."""
    with pytest.raises(ValueError, match="Input and output directories must be provided"):
        Settings(input_dir="", output_dir="/tmp/out")
    
    with pytest.raises(ValueError, match="Input and output directories must be provided"):
        Settings(input_dir="/tmp/in", output_dir="")


def test_configure_logging_plain_and_json(caplog):
    caplog.set_level(logging.INFO)
    configure_logging(level="INFO", fmt="plain")
    logging.getLogger("portfolio_advisor.test").info("hello")
    assert any("hello" in rec.message for rec in caplog.records)

    caplog.clear()
    configure_logging(level="INFO", fmt="json")
    logging.getLogger("portfolio_advisor.test").info("world")
    assert any("world" in rec.message for rec in caplog.records)


def test_configure_logging_library_mute_and_enable(caplog):
    caplog.set_level(logging.INFO)
    # By default, libraries should be muted (httpx in this case)
    configure_logging(level="INFO", fmt="plain", log_libraries=False)
    logging.getLogger("httpx").info("lib-muted")
    assert not any("lib-muted" in rec.message for rec in caplog.records)
    logging.getLogger("matplotlib").info("mpl-muted")
    assert not any("mpl-muted" in rec.message for rec in caplog.records)

    caplog.clear()
    # When enabled, they should pass through
    configure_logging(level="INFO", fmt="plain", log_libraries=True)
    logging.getLogger("httpx").info("lib-allowed")
    assert any("lib-allowed" in rec.message for rec in caplog.records)
    logging.getLogger("matplotlib").info("mpl-allowed")
    assert any("mpl-allowed" in rec.message for rec in caplog.records)

    caplog.clear()
    # Even when muted, warnings/errors should still appear
    configure_logging(level="INFO", fmt="plain", log_libraries=False)
    logging.getLogger("httpx").warning("lib-warning")
    assert any("lib-warning" in rec.message for rec in caplog.records)
    logging.getLogger("matplotlib").warning("mpl-warning")
    assert any("mpl-warning" in rec.message for rec in caplog.records)
