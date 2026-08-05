# auto-research-benchmark

Benchmark tasks + held-out grading, as a **library**. Two stable contracts:

| | contract | meaning |
|---|---|---|
| **Entry** | `Benchmark.load_task(id) -> Task` | what the problem is (agent-visible view only) |
| **Exit** | `Benchmark.grade(task, submission) -> Score` | grade a submission, ignorant of which system made it |

That asymmetry — the grader never knows who produced the submission — is what
lets grading stay fair. **Nothing here drives an agent**: the consuming system
(the thesis repo's loop) lists tasks, loads them, runs itself, and calls
`grade` at the end. (The pre-2026-07 adapter/runner/batch driving layers were
pruned; the agent loop, LLM client, traces, and grid scheduling all live in the
thesis repo now.)

## Architecture

```
arbench/
├── core/
│   ├── task.py           # Task: benchmark-agnostic unit of work (+ render_goal
│   │                     #   for container-visible paths)
│   ├── benchmark.py      # Benchmark ABC: list_tasks / load_task / grade
│   ├── result.py         # Score
│   └── data_version.py   # hash manifest of prepared dirs: stamped on first
│                         #   load, checked on every load, restamped in
│                         #   Score.details at grade (audit hook)
├── benchmarks/
│   ├── ale_bench/         # 40 competitive-programming tasks (self-contained grader)
│   └── openml_tabular/    # 101 curated tabular tasks (prepare + grade built in)
└── cli.py             # arbench tasks | arbench grade — the two hand-tools
```

A benchmark plugs in by implementing three methods — `list_tasks`,
`load_task` (the agent-visible Task only; never name the answers), and
`grade` (return `Score.invalid(...)`, never raise, on missing/malformed
submissions) — plus a `splits.yaml` (family + exclusions; roles are computed
from fractions, never stored) and a lazy-import branch in
`arbench.get_benchmark`. Library surface: `get_benchmark`,
`assign_roles/tasks_with_role/families/split_of` (lazy imports — the package
works on a bare machine). `Task.render_goal(visible_data_dir)` rewrites host
data paths for runs that see the data elsewhere (e.g. bind-mounted in a
container).

## Data layout + firewall

Data lives **inside the repo, gitignored** — re-downloadable input owned by the
plugin that prepares and grades it. Public and private are two distinct roots
so the answers firewall is mount-level (only the public roots are ever
container-bindable):

```
data/
├── ale/{public,private}/          # $ALE_DATA_DIR    -> public
└── openml/{public,private}/       # $OPENML_DATA_DIR -> public
                                   # private: <task>/answers.csv, never bound
```

No answers file exists under any public root — `tests/test_leakage.py` enforces
it. Task objects never name the private paths.

## Usage

```bash
uv sync --extra dev                                    # the library (tiny deps)
uv sync --extra ale-prep                               # + hf_hub for ALE problem zips

arbench tasks --benchmark openml_tabular
arbench grade --benchmark openml_tabular --task wine_quality submission.csv
arbench splits --fractions bank=0.5,val=0.2,test=0.3 --seed 0   # the role draw
```

```python
import arbench
bench = arbench.get_benchmark("openml_tabular")   # data dirs from env
task = bench.load_task("wine_quality")            # public view only
score = bench.grade(task, submission_path)        # Score(value, valid, ...)
```

### Preparing the benchmarks

`ale_bench` serves Sakana's ALE-Bench problems with a self-contained grader (no
Docker judge on any pipeline path). Fetch the problem data with
`uv sync --extra ale-prep && python scripts/prepare_ale_bench.py`; the dataset
is Xet-backed, so `hf_hub_download` does the download.

`openml_tabular` prepares itself over the public OpenML REST endpoints — no
extra and no credentials: `python -m arbench.benchmarks.openml_tabular.prepare`.

The library imports and tests cleanly with only `--extra dev`; a plugin's own
dependencies are needed only to actually prepare/grade its tasks.

## Tests

```bash
uv sync --extra dev
uv run pytest       # grading + leakage tests use tiny staged fakes; no GPU, no net
```
