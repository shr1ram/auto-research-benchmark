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
│   ├── task.py        # Task: benchmark-agnostic unit of work (+ render_goal
│   │                  #   for container-visible paths)
│   ├── benchmark.py   # Benchmark ABC: list_tasks / load_task / grade
│   └── result.py      # Score
├── benchmarks/
│   ├── mlebench_lite/     # wraps openai/mle-bench (vendor/mle-bench extra)
│   └── openml_tabular/    # 11 curated tabular tasks (prepare + grade built in)
├── batch/             # LEGACY: kept only until its scheduler semantics + tests
│                      #   are ported to the thesis grid driver, then deleted
└── cli.py             # arbench tasks | arbench grade — the two hand-tools
```

A benchmark plugs in by implementing three methods (`list_tasks`, `load_task`,
`grade`). Library surface: `arbench.get_benchmark(name)` (lazy imports — the
package works on a bare machine).

## Data layout + firewall

Data lives **inside the repo, gitignored** — re-downloadable input owned by the
plugin that prepares and grades it. Public and private are two distinct roots
so the answers firewall is mount-level (only the public roots are ever
container-bindable):

```
data/
├── mlebench/{public,private}/     # $MLEBENCH_DATA_DIR -> public
└── openml/{public,private}/       # $OPENML_DATA_DIR   -> public
                                   # private: <task>/answers.csv, never bound
```

No answers file exists under any public root — `tests/test_leakage.py` enforces
it. Task objects never name the private paths.

## Usage

```bash
uv sync --extra dev                                    # the library (tiny deps)
uv sync --extra mlebench                               # + vendor/mle-bench, run box only

arbench tasks --benchmark openml_tabular
arbench grade --benchmark openml_tabular --task wine_quality submission.csv
```

```python
import arbench
bench = arbench.get_benchmark("openml_tabular")   # data dirs from env
task = bench.load_task("wine_quality")            # public view only
score = bench.grade(task, submission_path)        # Score(value, valid, ...)
```

### Prerequisites for the MLE-Bench Lite benchmark

`mlebench_lite` wraps `openai/mle-bench`, so on the run box you need:

1. `uv sync --extra mlebench`  (editable dep on `vendor/mle-bench`)
2. Kaggle API creds at `~/.kaggle/kaggle.json`, **and** to accept each
   competition's rules on kaggle.com.
3. Prepared data: `mlebench prepare -c <competition>` (point its cache at the
   project filesystem, not NFS home).

The library imports and tests cleanly **without** mlebench installed — it's
only needed to actually prepare/grade those tasks.

## Tests

```bash
uv sync --extra dev
uv run pytest       # grading + leakage tests use tiny staged fakes; no GPU, no net
```
