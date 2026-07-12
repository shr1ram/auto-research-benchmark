"""Prepare ale_bench tasks: stage the PUBLIC tree from a grader-side
ale-bench session (needs the `ale` extra + a Docker host — run on a dev
machine, not the boxes).

    ALE_DATA_DIR=data/ale/public python scripts/prepare_ale_bench.py [ids...]
    # ids default to the official Lite ten; `--all` stages all 40;
    # `--full-seeds` stages the full public case set (50/problem) instead of
    # the lite five, and records seed_regime accordingly.

Stages per problem: problem.md (official statement), meta.json, the public
input cases under cases/, and the trust-on-first-use .data_version stamp.
NOTHING private is staged — hidden cases live only in ale-bench's own cache
and are materialised at private_eval time (see the plugin docstring's
FIREWALL note).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from arbench.benchmarks.ale_bench.tasks import ALL_PROBLEMS, LITE_PROBLEMS
from arbench.core.data_version import verify_data_version


def prepare_one(problem_id: str, root: Path, full_seeds: bool) -> None:
    import ale_bench
    session = ale_bench.start(problem_id=problem_id,
                              lite_version=not full_seeds,
                              run_visualization_server=False)
    try:
        prep = root / problem_id / "prepared"
        (prep / "cases").mkdir(parents=True, exist_ok=True)
        problem = session.problem
        (prep / "problem.md").write_text(problem.statement)
        seeds = list(session.public_seeds)
        cases = session.case_gen(seeds)
        if isinstance(cases, str):
            cases = [cases]
        for seed, case in zip(seeds, cases):
            (prep / "cases" / f"{seed:04d}.txt").write_text(case)
        meta = {"problem_id": problem_id,
                "score_type": str(problem.metadata.score_type.value),
                "judge_version": str(getattr(problem.metadata, "judge_version", "")),
                "time_limit_s": getattr(problem.constraints, "time_limit_s", None),
                "n_public_cases": len(seeds),
                "public_seeds": seeds,
                "seed_regime": "full" if full_seeds else "lite"}
        (prep / "meta.json").write_text(json.dumps(meta, indent=1))
        (prep / ".data_version").unlink(missing_ok=True)   # deliberate re-pin
        version = verify_data_version(prep)
        print(f"[prepare] {problem_id}: {len(seeds)} public cases "
              f"({meta['seed_regime']} regime) -> {prep} [{version}]")
    finally:
        session.close()


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
        raise SystemExit(f"unknown problem ids {unknown}; have {sorted(ALL_PROBLEMS)}")
    root = Path(os.environ.get("ALE_DATA_DIR", "data/ale/public"))
    for problem_id in ids:
        prepare_one(problem_id, root, full_seeds)


if __name__ == "__main__":
    main(sys.argv[1:])
