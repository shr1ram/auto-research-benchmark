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

def _stub_subprocess(monkeypatch, stdout="", returncode=0, calls=None):
    """Pretend every external tool exists and record what we shell out to."""
    monkeypatch.setattr(prepare.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, **kwargs):
        if calls is not None:
            calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout, "boom")

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)


# objdump prints this header for any file it successfully parses, including a
# static binary with no dynamic symbols at all.
_HEADER = "t:     file format elf64-x86-64\n\n"


def test_build_targets_musl_and_returns_that_bin_dir(tmp_path, monkeypatch):
    """The build must pass --target musl and read back the musl bin dir —
    a plain --release build lands in target/release and links glibc."""
    calls = []
    _stub_subprocess(monkeypatch, calls=calls)
    bin_dir = prepare._build_tools(tmp_path)

    build = next(c for c in calls if c[:2] == ["cargo", "build"])
    assert "--target" in build
    assert build[build.index("--target") + 1] == "x86_64-unknown-linux-musl"
    assert bin_dir == tmp_path / "target" / "x86_64-unknown-linux-musl" / "release"


def test_musl_target_is_installed_once_per_run_not_per_problem(monkeypatch):
    """`--all` builds 40 problems; the target is global toolchain state."""
    calls = []
    _stub_subprocess(monkeypatch, calls=calls)
    prepare._ensure_build_target()
    assert calls == [["rustup", "target", "add", "x86_64-unknown-linux-musl"]]


def test_glibc_linked_binary_is_refused(tmp_path, monkeypatch):
    """A dynamic build execs fine on the build box and dies on the cluster,
    so the failure must surface here, at stage time."""
    _stub_subprocess(monkeypatch, stdout=(
        _HEADER +
        "0000 DF *UND* 0000 (GLIBC_2.18) pthread_getname_np\n"
        "0000 DF *UND* 0000 (GLIBC_2.2.5) memcpy\n"))
    with pytest.raises(RuntimeError, match="dynamically linked against glibc"):
        prepare._assert_static(tmp_path / "tester")


def test_static_pie_binary_is_accepted(tmp_path, monkeypatch):
    """Static-PIE keeps a PT_DYNAMIC segment: objdump exits 0, lists nothing."""
    _stub_subprocess(monkeypatch, stdout=_HEADER)
    prepare._assert_static(tmp_path / "tester")     # must not raise


def test_static_non_pie_binary_is_accepted_despite_rc1(tmp_path, monkeypatch):
    """A fully static NON-PIE binary has no dynamic table at all, so objdump
    exits 1 ("not a dynamic object") after parsing the file happily. That is
    the strongest possible evidence of glibc independence and must PASS —
    keying the check on rc would falsely refuse it."""
    _stub_subprocess(monkeypatch, stdout=_HEADER, returncode=1)
    prepare._assert_static(tmp_path / "tester")     # must not raise


def test_failed_objdump_is_refused_not_read_as_clean(tmp_path, monkeypatch):
    """FAIL CLOSED: objdump erroring produces no GLIBC lines, which must not
    be mistaken for proof of a static binary. No header => never parsed."""
    _stub_subprocess(monkeypatch, stdout="", returncode=1)
    with pytest.raises(RuntimeError, match="could not verify static linkage"):
        prepare._assert_static(tmp_path / "tester")


def test_unparsable_output_is_refused_even_on_success(tmp_path, monkeypatch):
    """Exit 0 with no object header means objdump never parsed the file."""
    _stub_subprocess(monkeypatch, stdout="something unexpected\n")
    with pytest.raises(RuntimeError, match="could not verify static linkage"):
        prepare._assert_static(tmp_path / "tester")


def test_missing_objdump_is_an_error_not_a_silent_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="objdump not found"):
        prepare._assert_static(tmp_path / "tester")


def test_glibc_symbols_are_refused_even_when_objdump_exits_nonzero(
        tmp_path, monkeypatch):
    """The GLIBC scan is what enforces the guarantee; it must not be skipped
    just because the exit code was tolerated."""
    _stub_subprocess(monkeypatch, stdout=(
        _HEADER + "0000 DF *UND* 0000 (GLIBC_2.18) foo\n"), returncode=1)
    with pytest.raises(RuntimeError, match="dynamically linked against glibc"):
        prepare._assert_static(tmp_path / "tester")
