"""Tests for utils.fs module."""

from __future__ import annotations

import json
from pathlib import Path

from portfolio_advisor.utils.fs import utcnow_iso, write_json_atomic


def test_write_json_atomic(tmp_path: Path):
    """Test atomic JSON writing."""
    # Test basic write
    target = tmp_path / "subdir" / "data.json"
    data = {"key": "value", "number": 42}
    
    write_json_atomic(target, data)
    
    # Verify file exists and content is correct
    assert target.exists()
    with target.open("r") as fh:
        loaded = json.load(fh)
    assert loaded == data
    
    # Test overwrite
    new_data = {"updated": True}
    write_json_atomic(target, new_data)
    
    with target.open("r") as fh:
        loaded = json.load(fh)
    assert loaded == new_data
    
    # Ensure no .tmp files are left behind
    assert not any(p.suffix == ".tmp" for p in target.parent.iterdir())


def test_utcnow_iso():
    """Test UTC timestamp generation."""
    ts1 = utcnow_iso()
    
    # Should be in ISO format with timezone info
    assert "T" in ts1
    assert "+" in ts1 or "Z" in ts1 or ":" in ts1[-6:]  # Has timezone
    
    # Should not have microseconds
    time_part = ts1.split("T")[1].split("+")[0].split("Z")[0]
    assert "." not in time_part
    
    # Should be parseable
    from datetime import datetime
    # Handle both Z suffix and +00:00 format
    if ts1.endswith("Z"):
        parsed = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
    else:
        parsed = datetime.fromisoformat(ts1)
    assert parsed.tzinfo is not None
