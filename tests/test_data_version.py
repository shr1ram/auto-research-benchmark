"""data_version: stamp on first load, loud on drift, silent when stable."""
from __future__ import annotations

import pytest

from arbench.core.data_version import (
    STAMP_NAME, compute_data_version, verify_data_version,
)


def _tree(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.csv").write_text("a,b\n1,2\n")
    (root / "meta.json").write_text("{}")
    return root


def test_first_load_stamps_then_verifies(tmp_path):
    d = _tree(tmp_path / "prepared")
    v1 = verify_data_version(d)
    assert (d / STAMP_NAME).read_text().strip() == v1
    assert verify_data_version(d) == v1          # second load: quiet


def test_drift_fails_loudly(tmp_path):
    d = _tree(tmp_path / "prepared")
    verify_data_version(d)
    (d / "train.csv").write_text("a,b\n1,2\n3,4\n")   # size changed
    with pytest.raises(RuntimeError, match="data_version mismatch"):
        verify_data_version(d)


def test_added_file_changes_version_but_stamp_does_not(tmp_path):
    d = _tree(tmp_path / "prepared")
    v1 = compute_data_version(d)
    (d / STAMP_NAME).write_text("whatever\n")
    assert compute_data_version(d) == v1         # stamp file excluded
    (d / "extra.csv").write_text("x\n")
    assert compute_data_version(d) != v1


def test_openml_load_records_and_checks(tmp_path):
    pytest.importorskip("pandas")
    import json

    from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular

    prep = tmp_path / "public" / "wine_quality" / "prepared"
    prep.mkdir(parents=True)
    (prep / "meta.json").write_text(json.dumps(
        {"metric": "accuracy", "higher_better": True, "kind": "clf",
         "target": "y", "id_col": "id"}))
    (prep / "description.md").write_text("toy\n")
    (prep / "train.csv").write_text("id,y\n1,0\n")
    (prep / "sample_submission.csv").write_text("id,prediction\n2,0\n")
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"))
    task = bench.load_task("wine_quality")
    assert task.metadata["data_version"]
    # drift between loads fails the NEXT load loudly
    (prep / "train.csv").write_text("id,y\n1,0\n2,1\n")
    with pytest.raises(RuntimeError, match="data_version mismatch"):
        bench.load_task("wine_quality")
