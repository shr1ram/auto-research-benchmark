"""prepare_ale_bench.py: zip-slip guard + the static-link contract.

None of these need cargo or the network.
"""
from __future__ import annotations

import importlib.util
import subprocess
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


# --- the staged tools must be static: they are built on a newer-glibc lab box
# and consumed on an older-glibc cluster (see _build_tools). ------------------

def test_build_targets_musl_and_returns_that_bin_dir(tmp_path, monkeypatch):
    """The build must pass --target musl and read back the musl bin dir --
    a plain --release build lands in target/release and links glibc."""
    calls = []
    monkeypatch.setattr(prepare.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(prepare.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    bin_dir = prepare._build_tools(tmp_path)

    build = next(c for c in calls if c[:2] == ["cargo", "build"])
    assert "--target" in build
    assert build[build.index("--target") + 1] == "x86_64-unknown-linux-musl"
    assert bin_dir == tmp_path / "target" / "x86_64-unknown-linux-musl" / "release"


def test_build_installs_the_musl_target_first(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(prepare.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    prepare._build_tools(tmp_path)
    assert ["rustup", "target", "add", "x86_64-unknown-linux-musl"] in calls


def _objdump_stub(monkeypatch, output):
    monkeypatch.setattr(prepare.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        prepare.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, output, ""))


def test_glibc_linked_binary_is_refused(tmp_path, monkeypatch):
    """A dynamic build execs fine on the build box and dies on the cluster,
    so the failure must surface here, at stage time."""
    _objdump_stub(monkeypatch, (
        "DYNAMIC SYMBOL TABLE:\n"
        "0000 DF *UND* 0000 (GLIBC_2.18) pthread_getname_np\n"
        "0000 DF *UND* 0000 (GLIBC_2.2.5) memcpy\n"))
    with pytest.raises(RuntimeError, match="dynamically linked against glibc"):
        prepare._assert_static(tmp_path / "tester")


def test_static_musl_binary_is_accepted(tmp_path, monkeypatch):
    _objdump_stub(monkeypatch, "objdump: /x: not a dynamic object\n")
    prepare._assert_static(tmp_path / "tester")     # must not raise


def test_missing_objdump_is_an_error_not_a_silent_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="objdump not found"):
        prepare._assert_static(tmp_path / "tester")
