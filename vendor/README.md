# vendor/ — third-party source checkouts (gitignored)

Re-cloneable CODE lives here; downloaded DATA lives under `data/` (written by
each benchmark plugin's prepare step). Don't mix the two.

## mle-bench
- upstream: https://github.com/openai/mle-bench.git
- pinned:   507f92e1138bb6e40dac5c6ee7a6758e6424bf97 Update README.md (#143)
- used by:  `arbench.benchmarks.mlebench_lite` (registry + grade_csv; we never
  use its docker agent harness)
- install:  `uv sync --extra mlebench` (wired via [tool.uv.sources] as an
  editable path dependency)
- restore after a fresh clone: `git clone https://github.com/openai/mle-bench.git vendor/mle-bench && git -C vendor/mle-bench checkout 507f92e1138bb6e40dac5c6ee7a6758e6424bf97`
