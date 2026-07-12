"""ALE-Bench Lite task table: the official 10-problem Lite subset.

Provenance: problem ids from the dataset's own `problem_ids_lite.txt`
(HuggingFace SakanaAI/ALE-Bench, fetched 2026-07-12) — AtCoder Heuristic
Contest problems curated by ALE-Bench (Sakana AI, arXiv 2506.09050).
Ordering is alphabetical (contest number): footprints are near-identical
(statement + a handful of public cases), so SMALL_FIRST has nothing to sort.
"""
from __future__ import annotations

#: the official ALE-Bench Lite subset (problem_ids_lite.txt)
LITE_PROBLEMS = (
    "ahc008", "ahc011", "ahc015", "ahc016", "ahc024",
    "ahc025", "ahc026", "ahc027", "ahc039", "ahc046",
)
