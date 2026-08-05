"""Leakage firewall as an executable check (SoT: "beware of data leakage").

Fixture tests run anywhere; the live-tree sweeps run wherever the prepared data
is present (the box) and skip elsewhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular


def _toy_openml_tree(root: Path, tid: str = "MagicTelescope") -> tuple[Path, Path]:
    """Minimal split-layout openml tree using a REAL task id (load_task gates on
    the curated id set)."""
    prep = root / "public" / tid / "prepared"
    prep.mkdir(parents=True)
    (prep / "meta.json").write_text(json.dumps(
        {"metric": "accuracy", "higher_better": True, "kind": "clf",
         "target": "y", "id_col": "id"}))
    (prep / "description.md").write_text("Toy fixture task\n")
    (prep / "train.csv").write_text("id,y\n1,0\n")
    (prep / "sample_submission.csv").write_text("id,prediction\n2,0\n")
    priv = root / "private" / tid
    priv.mkdir(parents=True)
    (priv / "answers.csv").write_text("id,y\n2,1\n")
    return root / "public", root / "private"


def test_task_view_contains_no_answers(tmp_path):
    pub, priv = _toy_openml_tree(tmp_path)
    b = OpenMLTabular(data_dir=str(pub), private_data_dir=str(priv),
                      enforce_spec=False)
    t = b.load_task("MagicTelescope")
    # check against the actual private ROOT, not the substring "private" —
    # macOS tmp dirs live under /private/var and false-trip a substring check
    assert priv not in Path(t.data_dir).parents
    assert not list(Path(t.data_dir).rglob("answers.csv"))
    assert str(priv) not in t.goal and "answers" not in t.goal
    for v in t.metadata.values():
        assert str(priv) not in str(v)


def test_grader_reads_private_tree(tmp_path):
    pub, priv = _toy_openml_tree(tmp_path)
    b = OpenMLTabular(data_dir=str(pub), private_data_dir=str(priv),
                      enforce_spec=False)
    assert b._answers("MagicTelescope") == priv / "MagicTelescope" / "answers.csv"
    assert b._answers("MagicTelescope").exists()


def test_legacy_answers_in_prepared_refused(tmp_path):
    pub, priv = _toy_openml_tree(tmp_path)
    (pub / "MagicTelescope" / "prepared" / "answers.csv").write_text("id,y\n")
    b = OpenMLTabular(data_dir=str(pub), private_data_dir=str(priv),
                      enforce_spec=False)
    with pytest.raises(RuntimeError, match="FIREWALL"):
        b.load_task("MagicTelescope")


def test_render_goal_rewrites_host_paths(tmp_path):
    pub, priv = _toy_openml_tree(tmp_path)
    b = OpenMLTabular(data_dir=str(pub), private_data_dir=str(priv),
                      enforce_spec=False)
    t = b.load_task("MagicTelescope")
    rendered = t.render_goal("/data")
    assert str(t.data_dir) not in rendered
    assert "Read the training data under: /data" in rendered
    # paths UNDER data_dir (sample submission) are rewritten too
    assert f"/data/sample_submission.csv" in rendered or "sample_submission" not in rendered


def test_render_goal_no_data_dir_is_noop():
    from arbench.core.task import Task
    t = Task(task_id="x", benchmark="b", goal="do things", eval="acc")
    assert t.render_goal("/data") == "do things"


# ── live-tree sweeps (box only; skip without prepared data) ──────────────────

_OPENML = os.environ.get("OPENML_DATA_DIR", "")


@pytest.mark.skipif(not _OPENML, reason="OPENML_DATA_DIR not set (live data check)")
def test_live_openml_public_root_has_zero_answer_files():
    root = Path(_OPENML)
    assert root.exists()
    assert not list(root.rglob("answers.csv")), "answers leaked into the public tree"


@pytest.mark.skipif(not _OPENML, reason="OPENML_DATA_DIR not set (live data check)")
def test_live_openml_served_tasks_are_clean():
    b = OpenMLTabular(enforce_spec=False)
    for tid in b.list_tasks():
        if not (b._prepared(tid) / "meta.json").exists():
            continue
        t = b.load_task(tid)
        assert "private" not in Path(t.data_dir).parts
        assert not list(Path(t.data_dir).rglob("answers.csv"))
