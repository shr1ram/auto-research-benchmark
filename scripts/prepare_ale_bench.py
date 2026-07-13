"""Prepare ale_bench tasks — SELF-CONTAINED, runs on the lab boxes.

    ALE_DATA_DIR=data/ale/public python scripts/prepare_ale_bench.py [ids...]
    # ids default to the official Lite ten; --all stages all 40;
    # --full-seeds uses the full public/private seed lists instead of the
    # official lite lists (50/2000+ cases vs 5/~10%).

Needs only: network (HuggingFace zip download) + cargo (a user-level
`rustup` install — no root, no Docker). Per problem:

  1. download <pid>.zip from HF SakanaAI/ALE-Bench into the PRIVATE root
     (`_zips/`) — data.json inside carries the PRIVATE seed lists, so the
     zip itself is grader-only material;
  2. `cargo build --release` the official tools (gen / vis / tester);
  3. `gen` the public cases -> PUBLIC tree, private cases -> PRIVATE tree;
  4. stage statement + meta publicly, tool binaries + private seeds
     privately; stamp .data_version (a re-run is a deliberate re-pin).

The public tree never receives private seeds or private cases; meta.json
carries public seeds and counts only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS, LITE_PROBLEMS
from arbench.core.data_version import verify_data_version

HF_URL = "https://huggingface.co/datasets/SakanaAI/ALE-Bench/resolve/main/{pid}.zip"
TOOL_BINARIES = ("gen", "vis", "tester")


def _fetch_zip(pid: str, zips_dir: Path) -> Path:
    zips_dir.mkdir(parents=True, exist_ok=True)
    dest = zips_dir / f"{pid}.zip"
    if not dest.exists():
        print(f"[prepare] {pid}: downloading zip")
        urllib.request.urlretrieve(HF_URL.format(pid=pid), dest)
    return dest


def _build_tools(tools_dir: Path) -> Path:
    """cargo build --release; returns the target bin dir."""
    if shutil.which("cargo") is None:
        raise SystemExit(
            "cargo not found — install a user-level Rust toolchain first:\n"
            "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y   (no root needed)")
    subprocess.run(["cargo", "build", "--release", "--quiet"],
                   cwd=tools_dir, check=True)
    return tools_dir / "target" / "release"


def _gen_cases(gen_binary: Path, seeds: list[int], workdir: Path) -> list[Path]:
    """Run the official generator: writes in/<i>.txt in seed-list order."""
    seeds_file = workdir / "seeds.txt"
    seeds_file.write_text("\n".join(str(s) for s in seeds) + "\n")
    subprocess.run([str(gen_binary), str(seeds_file)], cwd=workdir, check=True)
    produced = sorted((workdir / "in").glob("*.txt"))
    if len(produced) != len(seeds):
        raise RuntimeError(f"generator produced {len(produced)} cases for "
                           f"{len(seeds)} seeds")
    return produced


def _stage_cases(produced: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(produced):
        shutil.copy(src, dest / f"case_{i:04d}.txt")


def prepare_one(pid: str, public_root: Path, private_root: Path,
                full_seeds: bool) -> None:
    zip_path = _fetch_zip(pid, private_root / "_zips")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work)
        src = work / pid
        data = json.loads((src / "data.json").read_text())
        seeds = data["seeds"]
        regime = "full" if full_seeds else "lite"
        public_seeds = seeds["public" if full_seeds else "public_lite"]
        private_seeds = seeds["private" if full_seeds else "private_lite"]

        bin_dir = _build_tools(src / "tools")
        gen = bin_dir / "gen"

        # cases: generator runs in throwaway workdirs (it writes ./in/)
        pub_work, priv_work = work / "pub", work / "priv"
        pub_work.mkdir()
        priv_work.mkdir()
        prep = public_root / pid / "prepared"
        _stage_cases(_gen_cases(gen, public_seeds, pub_work), prep / "cases")
        priv = private_root / pid
        _stage_cases(_gen_cases(gen, private_seeds, priv_work),
                     priv / "cases")

        # official tools -> private bin (grader-only; vis scores, tester is
        # the future reactive path)
        (priv / "bin").mkdir(parents=True, exist_ok=True)
        for name in TOOL_BINARIES:
            built = bin_dir / name
            if built.exists():
                shutil.copy2(built, priv / "bin" / name)
        (priv / "seeds.json").write_text(json.dumps(
            {"regime": regime, "private_seeds": private_seeds}))

        # public view: statement + meta (public seeds ONLY)
        shutil.copy(src / "statement_en.md", prep / "problem.md")
        meta = {"problem_id": pid,
                "problem_type": data["metadata"]["problem_type"],
                "score_type": data["metadata"]["score_type"],
                "time_limit_s": data["constraints"].get("time_limit"),
                "memory_limit": data["constraints"].get("memory_limit"),
                "contest_start_at": data["metadata"].get("start_at"),
                "n_public_cases": len(public_seeds),
                "n_private_cases": len(private_seeds),
                "public_seeds": public_seeds,
                "seed_regime": regime}
        (prep / "meta.json").write_text(json.dumps(meta, indent=1))
        (prep / ".data_version").unlink(missing_ok=True)   # deliberate re-pin
        version = verify_data_version(prep)
        print(f"[prepare] {pid}: {data['metadata']['problem_type']}, "
              f"{len(public_seeds)} public / {len(private_seeds)} private "
              f"cases ({regime}) -> {prep} [{version}]")


def main(argv: list[str]) -> None:
    full_seeds = "--full-seeds" in argv
    stage_all = "--all" in argv
    ids = [a for a in argv if not a.startswith("--")]
    if stage_all and ids:
        raise SystemExit("pass either --all or explicit ids, not both")
    if stage_all:
        ids = list(ALL_PROBLEMS)
    elif not ids:
        ids = list(LITE_PROBLEMS)
    unknown = sorted(set(ids) - set(ALL_PROBLEMS))
    if unknown:
        raise SystemExit(f"unknown ids {unknown}; have {sorted(ALL_PROBLEMS)}")
    public_root = Path(os.environ.get("ALE_DATA_DIR", "data/ale/public"))
    private_root = Path(os.environ.get("ALE_PRIVATE_DATA_DIR",
                                       str(public_root.parent / "private")))
    for pid in ids:
        prepare_one(pid, public_root, private_root, full_seeds)


if __name__ == "__main__":
    main(sys.argv[1:])
