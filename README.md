# auto-research-benchmark

A small, system-agnostic harness for running **arbitrary autoresearch systems**
against **arbitrary benchmarks**, behind two stable contracts:

| | contract | meaning |
|---|---|---|
| **Entry** | `AutoResearchAdapter.run(task, workspace) -> submission` | drive a system (AIDE, continual-auto-research, …) to produce a submission |
| **Exit** | `Benchmark.grade(task, submission) -> Score` | grade that submission, ignorant of which system made it |

That asymmetry — the grader never knows who produced the submission — is what
lets you compare systems fairly on the same task.

First benchmark wired up: **MLE-Bench Lite** (OpenAI's
[mle-bench](https://github.com/openai/mle-bench) low-complexity split).
First system wired up: **[AIDE](https://github.com/WecoAI/aideml)**.

## Architecture

```
arbench/
├── core/
│   ├── task.py        # Task: benchmark-agnostic unit of work
│   ├── adapter.py     # AutoResearchAdapter  (ENTRY contract, ABC)
│   ├── benchmark.py   # Benchmark            (EXIT contract, ABC)
│   ├── result.py      # Score, RunResult
│   ├── runner.py      # load_task -> prepare -> run -> grade
│   └── registry.py    # name -> adapter / benchmark
├── adapters/
│   ├── aide_adapter.py        # drives forked AIDE
│   └── continual_adapter.py   # continual-auto-research (stub)
├── benchmarks/
│   └── mlebench_lite/         # wraps openai/mle-bench (registry + grade_csv)
└── llm/
    └── backends.py    # pluggable LLM: litellm(Kimi) | claude_p | local(Ollama)
```

A system plugs in by implementing one method (`run`). A benchmark plugs in by
implementing three (`list_tasks`, `load_task`, `grade`). Nothing else changes.

## LLM backends

AIDE (and most OpenAI-API clients) route to an OpenAI-compatible endpoint when
`OPENAI_BASE_URL` is set and the model name isn't a `gpt-`/`claude-`/`gemini-`
prefix. We exploit that to make the backend pluggable:

- `litellm` — team LiteLLM proxy (`litellm.yangtzeailab.com/v1`); **default model
  `Kimi-K2.6`** (also serves `deepseek-v4-flash`). Key from `CUSTOM_API_KEY`.
- `claude_p` — Claude via an OpenAI-compatible gateway.
- `local` — local Ollama on the UCL GPU box (`127.0.0.1:11435/v1`).

## Usage

```bash
uv pip install -e .                 # the harness itself (tiny deps)

arbench list                        # show registered adapters + benchmarks

# Run AIDE on one MLE-Bench Lite competition (on the GPU box):
arbench run \
  --adapter aide --benchmark mlebench_lite \
  --task random-acts-of-pizza \
  --backend litellm                 # -> Kimi-K2.6 \
  --steps 10 \
  --data-dir "$MLEBENCH_DATA_DIR" \
  --workspace "$PROJECT_FS/runs/raop" \
  --out result.json
```

### Prerequisites for the MLE-Bench Lite benchmark

`mlebench_lite` wraps `openai/mle-bench`, so on the run box you need:

1. `uv pip install -e /path/to/mle-bench`
2. Kaggle API creds at `~/.kaggle/kaggle.json`, **and** to accept each
   competition's rules on kaggle.com.
3. Prepared data: `mlebench prepare -c <competition>` (point its cache at the
   project filesystem, not NFS home).

The harness itself imports and tests cleanly **without** AIDE or mlebench
installed — those are only needed to actually run/grade.

## Parallel sweeps across GPU boxes (`arbench batch`)

Runs many `(task × seed × arm)` jobs concurrently across GPU boxes — each one a
full traced bundle. Self-contained: it discovers free boxes (ssh `nvidia-smi`),
leases them atomically on the shared FS (no double-claims), dispatches
`arbench run` over ssh, tracks `manifest.json`, releases boxes as jobs finish,
and is **resumable** (a job whose `run.json` exists is skipped).

```bash
arbench batch \
  --adapter aide --benchmark mlebench_lite \
  --tasks all-lite            # or a,b,c  or @file \
  --seeds 3                   # independent repeats -> variance/error bars \
  --arms baseline             # later: baseline,continual \
  --max-boxes 8 \
  --steps 8 --backend litellm --model Kimi-K2.6 \
  --data-dir "$MLEBENCH_DATA_DIR" \
  --venv  "$ROOT/.venv/bin/activate"   # shared-FS venv each box sources \
  --repo-dir "$ROOT" \
  --out runs/sweep-001
```

Each job lands in `runs/sweep-001/<arm>/<task>-seed<n>/run.json`. The scheduler
caches box discovery (spares the ssh jump host) and kills+frees any job exceeding
`--job-timeout`. For long sweeps, launch **detached on a lab box** so it survives
your laptop disconnecting:

```bash
# on a lab box that can ssh siblings:
scripts/run_sweep.sh baseline-v1 all-lite 3 8 8     # name tasks seeds max-boxes steps
tail -f runs/baseline-v1/sweep.log
```

Re-running the same sweep name resumes it (done jobs skipped).

## Tests

```bash
uv pip install -e ".[dev]"
pytest                 # contract + backend tests use fakes, no heavy deps
```
