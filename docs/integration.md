# Integration notes

## How a new autoresearch system plugs in

Implement `AutoResearchAdapter.run(task, workspace) -> submission_path`:

- read `task.goal`, `task.eval`, `task.data_dir`;
- do whatever the system does;
- write the submission to `task.submission_path(workspace)` and return it.

Register it with `register_adapter("name", Factory)`. That's the whole contract —
the runner and every benchmark already work with it. See `adapters/continual_adapter.py`
for a full example and `adapters/continual_adapter.py` for the stub shape.

## How a new benchmark plugs in

Implement `Benchmark`:

- `list_tasks()` → ids;
- `load_task(id)` → a `Task` (prepare/locate data, write the goal/eval);
- `grade(task, submission_path)` → a `Score` (return `Score.invalid(...)`, never
  raise, on missing/malformed submissions).

Register with `register_benchmark("name", Factory)`.

## MLE-Bench Lite specifics

- "Lite" = mle-bench's **low-complexity split** (22 Kaggle competitions, ~158 GB
  total). We list them smallest-first (`SMALL_FIRST`) so the first end-to-end run
  is cheap — start with `random-acts-of-pizza` (tiny text classification).
- We wrap mle-bench's own `registry` + `grade_csv`; we don't reimplement grading.
- **Kaggle gate:** `mlebench prepare` downloads via the Kaggle API and needs
  `~/.kaggle/kaggle.json` plus accepting each competition's rules on kaggle.com.
  This is a manual, per-account step and the main blocker to a fully automated run.
- **No docker required.** mle-bench ships a docker-based reference agent harness;
  we don't use it — we run the mle-bench *library* (prepare/grade) and the adapter
  natively, which is enough for the entry/exit contracts.

## LLM routing

Clients route to an OpenAI-compatible endpoint when `OPENAI_BASE_URL` is set and the
model name has no `gpt-`/`claude-`/`gemini-` prefix. `Kimi-K2.6` qualifies, so
`configure_llm_env("litellm")` (sets `OPENAI_BASE_URL` + `OPENAI_API_KEY`) is all
the client needs. Backends: `litellm` (Kimi, default), `claude_p`, `local` (Ollama).

## UCL box conventions

- Run on a GPU box (chosen: an idle lab-gpu box); keep **all** data/artifacts on
  the project FS `/cs/student/project_msc/.../sruppage` — NFS home is full.
- Set `MLEBENCH_DATA_DIR` and `XDG_CACHE_HOME` onto the project FS before prepare.
