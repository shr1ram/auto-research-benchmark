# Integration notes

## How a consuming system uses the library

```python
import arbench
bench = arbench.get_benchmark("openml_tabular")        # or mlebench_lite
for task_id in bench.list_tasks(): ...
task = bench.load_task(task_id)     # goal/eval prose + public data_dir only
# ... the system does its work, writes task.submission_path(workspace) ...
score = bench.grade(task, submission_path)
```

`Task.render_goal(visible_data_dir)` rewrites host data paths for runs that see
the data elsewhere (e.g. bind-mounted inside a Singularity container).

## How a new benchmark plugs in

Implement `Benchmark`:

- `list_tasks()` → ids;
- `load_task(id)` → a `Task` (prepare/locate data, write the goal/eval; never
  name the private answers);
- `grade(task, submission_path)` → a `Score` (return `Score.invalid(...)`, never
  raise, on missing/malformed submissions).

Add its lazy-import branch to `arbench.get_benchmark` and its public/private
roots under `data/<name>/{public,private}`.

## MLE-Bench Lite specifics

- "Lite" = mle-bench's **low-complexity split** (22 Kaggle competitions, ~158 GB
  total). We list them smallest-first (`SMALL_FIRST`) so the first end-to-end run
  is cheap — start with `random-acts-of-pizza` (tiny text classification).
- We wrap mle-bench's own `registry` + `grade_csv`; we don't reimplement grading.
- **Kaggle gate:** `mlebench prepare` downloads via the Kaggle API and needs
  `~/.kaggle/kaggle.json` plus accepting each competition's rules on kaggle.com.
  This is a manual, per-account step and the main blocker to a fully automated run.
- **No docker required.** mle-bench ships a docker-based reference agent harness;
  we don't use it — we run the mle-bench *library* (prepare/grade) natively,
  which is enough for the entry/exit contracts.

## UCL box conventions

- Run on a GPU box (chosen: an idle lab-gpu box); keep **all** data/artifacts on
  the project FS `/cs/student/project_msc/.../sruppage` — NFS home is full.
- Set `MLEBENCH_DATA_DIR` / `OPENML_DATA_DIR` (the PUBLIC roots under
  `data/`) and `XDG_CACHE_HOME` onto the project FS before prepare.
