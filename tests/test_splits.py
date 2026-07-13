"""Automatic splits: facts in the yaml, roles computed from fractions."""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from arbench.benchmarks.mlebench_lite.tasks import LITE_COMPETITIONS
from arbench.benchmarks.openml_tabular.tasks import BY_ID
from arbench.cli import main
from arbench.core.splits import (
    ROLES, assign_roles, families, load_split_meta, split_of, tasks_with_role,
)

FR = {"bank": 0.5, "val": 0.2, "test": 0.3}


def test_every_task_is_listed_with_a_family():
    assert set(load_split_meta("openml_tabular")) == set(BY_ID)
    assert set(load_split_meta("mlebench_lite")) == set(LITE_COMPETITIONS)


def test_families_exclude_the_excluded_with_reasons():
    fams = families()
    flat = {t for members in fams.values() for t in members}
    meta = load_split_meta("mlebench_lite")
    for task_id, entry in meta.items():
        if entry.get("excluded"):
            assert ("mlebench_lite", task_id) not in flat
            assert entry["excluded"]              # reason is mandatory
        else:
            assert ("mlebench_lite", task_id) in flat


def test_families_are_yaml_defined_not_hardcoded():
    """Add/remove families by editing the yaml — the code derives everything."""
    fams = families()
    assert set(fams) == {"tabular", "text", "vision", "other"}
    benchmarks_in_tabular = {b for b, _ in fams["tabular"]}
    assert benchmarks_in_tabular == {"openml_tabular", "mlebench_lite"}
    assert fams["other"] == [("mlebench_lite", "mlsp-2013-birds")]


def test_assignment_is_deterministic_and_complete():
    a = assign_roles(FR, seed=0)
    b = assign_roles(FR, seed=0)
    assert a == b                                  # pure in (fractions, seed)
    assert set(a.values()) <= set(ROLES)
    flat = {t for members in families().values() for t in members}
    assert set(a) == flat                          # every drawable task, once
    assert assign_roles(FR, seed=1) != a           # seed moves the draw


def test_stratified_largest_remainder_counts():
    a = assign_roles(FR, seed=0)
    for fam, members in families().items():
        n = len(members)
        counts = {r: sum(1 for m in members if a[m] == r) for r in ROLES}
        assert sum(counts.values()) == n
        for role, frac in FR.items():
            assert abs(counts[role] - n * frac) < 1, (fam, role, counts)


def test_fraction_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        assign_roles({"bank": 0.6, "val": 0.2, "test": 0.3}, 0)
    with pytest.raises(ValueError, match="exactly"):
        assign_roles({"bank": 0.5, "train": 0.2, "val": 0.1, "test": 0.2}, 0)
    with pytest.raises(ValueError, match="unknown role"):
        tasks_with_role("holdout", FR, 0)


def test_tasks_with_role_filters():
    test_openml = tasks_with_role("test", FR, 0, benchmark="openml_tabular")
    assert test_openml
    assert set(test_openml) <= set(BY_ID)
    vision_bank = tasks_with_role("bank", FR, 0, family="vision")
    assert all(t in LITE_COMPETITIONS for t in vision_bank)


def test_split_of_returns_facts():
    entry = split_of("openml_tabular", "wine_quality")
    assert entry["family"] == "tabular"
    assert split_of("openml_tabular", "nope") is None


def test_cli_families_and_assignment():
    bare = CliRunner().invoke(main, ["splits"])
    assert bare.exit_code == 0
    assert "tabular (" in bare.output
    assert "excluded: the-icml-2013-whale-challenge-right-whale-redux" in bare.output
    drawn = CliRunner().invoke(
        main, ["splits", "--fractions", "bank=0.5,val=0.2,test=0.3"])
    assert drawn.exit_code == 0
    assert "bank " in drawn.output and "test " in drawn.output


def test_loaded_task_carries_split_facts(tmp_path):
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
                          private_data_dir=str(tmp_path / "private"),
                          enforce_spec=False)
    task = bench.load_task("wine_quality")
    assert task.metadata["split"]["family"] == "tabular"
