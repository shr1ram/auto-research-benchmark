"""Prepare ale_bench_lite tasks: stage the PUBLIC tree from a grader-side
ale-bench session (needs the `ale` extra + a Docker host — run on a dev
machine, not the boxes).

    ALE_DATA_DIR=data/ale/public python scripts/prepare_ale_bench.py [ids...]

Stages per problem: problem.md (official statement), meta.json, and the Lite
public input cases under cases/. NOTHING private is staged — hidden cases
live only in ale-bench's own cache and are materialised at private_eval time
(see the plugin docstring's FIREWALL note).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from arbench.benchmarks.ale_bench_lite.tasks import LITE_PROBLEMS


def prepare_one(problem_id: str, root: Path) -> None:
    import ale_bench
    session = ale_bench.start(problem_id=problem_id, lite_version=True,
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
                "public_seeds": seeds}
        (prep / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[prepare] {problem_id}: {len(seeds)} public cases -> {prep}")
    finally:
        session.close()


def main(argv: list[str]) -> None:
    root = Path(os.environ.get("ALE_DATA_DIR", "data/ale/public"))
    ids = argv or list(LITE_PROBLEMS)
    unknown = sorted(set(ids) - set(LITE_PROBLEMS))
    if unknown:
        raise SystemExit(f"unknown problem ids {unknown}; have {sorted(LITE_PROBLEMS)}")
    for problem_id in ids:
        prepare_one(problem_id, root)


if __name__ == "__main__":
    main(sys.argv[1:])
