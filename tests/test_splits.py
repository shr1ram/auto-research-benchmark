"""Splits index: complete, disjoint, real task ids, design invariants."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from arbench.benchmarks.mlebench_lite.tasks import LITE_COMPETITIONS
from arbench.benchmarks.openml_tabular.tasks import BY_ID
from arbench.cli import main
from arbench.core.splits import load_splits, split_index, split_of, tasks_in


def test_every_openml_task_has_exactly_one_bin():
    table = load_splits("openml_tabular")
    assert set(table) == set(BY_ID)          # complete, no strays


def test_every_lite_competition_has_exactly_one_bin():
    table = load_splits("mlebench_lite")
    assert set(table) == set(LITE_COMPETITIONS)


def test_in_domain_split_shape():
    """The seeded random 7/3 split, frozen: eval must contain both task kinds
    (stratification), pol is the measured-saturation control."""
    source = tasks_in("in_domain_source")
    eval_ = tasks_in("in_domain_eval")
    assert len(source) == 7 and len(eval_) == 3
    kinds = {load_splits("openml_tabular")[t]["kind"] for t in eval_}
    assert kinds == {"classification", "regression"}
    assert tasks_in("control") == ["pol"]


def test_near_bin_is_mlebench_tabular_only():
    table = load_splits("mlebench_lite")
    for task_id in tasks_in("near"):
        assert table[task_id]["modality"] == "tabular"
    assert len(tasks_in("near")) == 4


def test_far_bins_are_modality_pure():
    table = load_splits("mlebench_lite")
    assert all(table[t]["modality"] == "vision" for t in tasks_in("far_vision"))
    assert all(table[t]["modality"] == "text" for t in tasks_in("far_text"))


def test_excluded_tasks_carry_reasons():
    table = load_splits("mlebench_lite")
    for task_id in tasks_in("excluded"):
        assert table[task_id].get("note"), task_id


def test_index_covers_everything_once():
    index = split_index()
    flat = [t for entries in index.values() for t in entries]
    assert len(flat) == len(set(flat)) == len(BY_ID) + len(LITE_COMPETITIONS)


def test_split_of_and_unknown_bin():
    assert split_of("openml_tabular", "wine_quality")["bin"] == "in_domain_source"
    assert split_of("openml_tabular", "nope") is None
    with pytest.raises(ValueError, match="unknown bin"):
        tasks_in("sideways")


def test_cli_splits_command():
    result = CliRunner().invoke(main, ["splits", "--bin", "near"])
    assert result.exit_code == 0
    assert "leaf-classification" in result.output


def test_loaded_task_carries_its_split(tmp_path):
    pd = pytest.importorskip("pandas")
    import json

    from arbench.benchmarks.openml_tabular.benchmark import OpenMLTabular

    prep = tmp_path / "public" / "wine_quality" / "prepared"
    prep.mkdir(parents=True)
    (prep / "meta.json").write_text(json.dumps(
        {"metric": "rmse", "higher_better": False, "kind": "regression",
         "target": "y", "id_col": "id"}))
    (prep / "description.md").write_text("toy\n")
    (prep / "train.csv").write_text("id,y\n1,0\n")
    (prep / "sample_submission.csv").write_text("id,prediction\n2,0\n")
    bench = OpenMLTabular(data_dir=str(tmp_path / "public"),
                          private_data_dir=str(tmp_path / "private"))
    task = bench.load_task("wine_quality")
    assert task.metadata["split"]["bin"] == "in_domain_source"
    assert task.metadata["split"]["modality"] == "tabular"
