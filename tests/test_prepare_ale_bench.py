"""prepare_ale_bench.py: the zip-slip guard (no cargo/network needed)."""
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_ale_bench", REPO / "scripts" / "prepare_ale_bench.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_zip_slip_member_is_refused(tmp_path, monkeypatch):
    evil = tmp_path / "ahc008.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("ahc008/data.json", "{}")
        zf.writestr("../escape.txt", "pwned")          # zip-slip
    monkeypatch.setattr(prepare, "_fetch_zip", lambda pid, zips_dir: evil)
    monkeypatch.setattr(prepare, "_build_tools",
                        lambda tools_dir: tmp_path / "nope")
    with pytest.raises(RuntimeError, match="unsafe member"):
        prepare.prepare_one("ahc008", tmp_path / "pub", tmp_path / "priv",
                            full_seeds=False)
    assert not (tmp_path / "escape.txt").exists()
