"""Splits index: mechanics always tested; design invariants arm themselves
the moment the human signs off assignments (until then everything ships
`unassigned` and the invariant tests skip)."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from arbench.benchmarks.mlebench_lite.tasks import LITE_COMPETITIONS
from arbench.benchmarks.openml_tabular.tasks import BY_ID
from arbench.cli import main
from arbench.core.splits import load_splits, split_index, split_of, tasks_in


def _all_unassigned() -> bool:
    return not any(e["bin"] != "unassigned"
                   for b in ("openml_tabular", "mlebench_lite")
                   for e in load_splits(b).values())

assigned = pytest.mark.skipif(
    _all_unassigned(), reason="split assignments await explicit human sign-off")


# ------------------------------------------------------ mechanics (always)

def test_every_openml_task_is_listed():
    assert set(load_splits("openml_tabular")) == set(BY_ID)


def test_every_lite_competition_is_listed():
    assert set(load_splits("mlebench_lite")) == set(LITE_COMPETITIONS)


def test_index_covers_everything_once():
    index = split_index()
    flat = [t for entries in index.values() for t in entries]
    assert len(flat) == len(set(flat)) == len(BY_ID) + len(LITE_COMPETITIONS)


def test_split_of_and_unknown_bin():
    assert split_of("openml_tabular", "wine_quality") is not None
    assert split_of("openml_tabular", "nope") is None
    with pytest.raises(ValueError, match="unknown bin"):
        tasks_in("sideways")


def test_unassigned_bins_run_nothing():
    """The sign-off gate: until assignment, every task is unassigned and the
    runnable bins are all EMPTY — a split: config ref cannot dispatch."""
    if not _all_unassigned():
        pytest.skip("assignments signed off — gate no longer applicable")
    for bin_name in ("in_domain_source", "in_domain_eval", "near",
                     "far_vision", "far_text"):
        assert tasks_in(bin_name) == []


def test_cli_splits_command():
    result = CliRunner().invoke(main, ["splits", "--bin", "unassigned"])
    assert result.exit_code == 0
    assert "wine_quality" in result.output


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
    assert task.metadata["split"]["bin"] in ("unassigned", "in_domain_source")


# ------------------------------- design invariants (armed on sign-off)

@assigned
def test_in_domain_split_shape():
    source, eval_ = tasks_in("in_domain_source"), tasks_in("in_domain_eval")
    assert len(source) == 7 and len(eval_) == 3
    kinds = {load_splits("openml_tabular")[t]["kind"] for t in eval_}
    assert kinds == {"classification", "regression"}
    assert tasks_in("control") == ["pol"]


@assigned
def test_near_bin_is_mlebench_tabular_only():
    table = load_splits("mlebench_lite")
    assert all(table[t]["modality"] == "tabular" for t in tasks_in("near"))


@assigned
def test_far_bins_are_modality_pure():
    table = load_splits("mlebench_lite")
    assert all(table[t]["modality"] == "vision" for t in tasks_in("far_vision"))
    assert all(table[t]["modality"] == "text" for t in tasks_in("far_text"))


@assigned
def test_excluded_tasks_carry_reasons():
    table = load_splits("mlebench_lite")
    for task_id in tasks_in("excluded"):
        assert table[task_id].get("note"), task_id
