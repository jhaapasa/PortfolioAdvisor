from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_advisor.errors import InputOutputError
from portfolio_advisor.io_utils import list_input_files, read_files_preview, write_output_text


def test_list_and_preview_io(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    f1 = in_dir / "a.txt"
    f1.write_text("hello", encoding="utf-8")
    f2 = in_dir / "b.txt"
    f2.write_text("world", encoding="utf-8")

    files = list_input_files(str(in_dir))
    assert [p.name for p in files] == ["a.txt", "b.txt"]

    previews = read_files_preview(files, max_bytes=3)
    assert previews and previews[0][1]

    out_path = write_output_text(str(out_dir), "res.txt", "ok")
    assert Path(out_path).exists()


def test_list_input_files_missing_dir(tmp_path: Path):
    with pytest.raises(InputOutputError):
        list_input_files(str(tmp_path / "missing"))
