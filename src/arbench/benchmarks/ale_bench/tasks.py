"""ALE-Bench task table: the full 40-problem set + the official Lite subset.

Provenance: problem ids from the dataset's own `problem_ids.txt` /
`problem_ids_lite.txt` (HuggingFace SakanaAI/ALE-Bench, fetched 2026-07-12)
— AtCoder Heuristic Contest problems (plus two sponsored contests) curated
by ALE-Bench (Sakana AI, arXiv 2506.09050). ALL_PROBLEMS keeps the dataset
file's own order; footprints are near-identical (statement + a handful of
public cases), so SMALL_FIRST has nothing to sort.
"""
from __future__ import annotations

#: the full ALE-Bench problem set (problem_ids.txt, dataset order)
ALL_PROBLEMS = (
    "ahc001", "ahc002", "ahc003", "ahc004", "ahc005",
    "future-contest-2022-qual",
    "ahc006", "ahc007", "ahc008", "ahc009", "ahc010", "ahc011", "ahc012",
    "ahc014", "ahc015", "ahc016", "ahc017", "ahc019", "ahc020", "ahc021",
    "toyota2023summer-final",
    "ahc024", "ahc025", "ahc026", "ahc027", "ahc028", "ahc030", "ahc031",
    "ahc032", "ahc033", "ahc034", "ahc035", "ahc038", "ahc039", "ahc040",
    "ahc041", "ahc042", "ahc044", "ahc045", "ahc046",
)

#: the official Lite subset (problem_ids_lite.txt) — the cheap starter ten
LITE_PROBLEMS = (
    "ahc008", "ahc011", "ahc015", "ahc016", "ahc024",
    "ahc025", "ahc026", "ahc027", "ahc039", "ahc046",
)

assert set(LITE_PROBLEMS) <= set(ALL_PROBLEMS)
