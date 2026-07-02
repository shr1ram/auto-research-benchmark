#!/usr/bin/env bash
# Launch an arbench batch sweep DETACHED on a lab box, so it survives your laptop
# disconnecting. Run this ON a lab box (e.g. eider) that can ssh sibling boxes.
#
# Usage (on the box):
#   scripts/run_sweep.sh <sweep-name> "<task,list|all-lite>" <seeds> <max-boxes> <steps>
# e.g.
#   scripts/run_sweep.sh baseline-v1 all-lite 3 8 8
#
# Resumable: re-run with the same <sweep-name> to continue (done jobs are skipped).
set -euo pipefail

PROJ="${PROJECT_FS:-/cs/student/project_msc/2025/csml/sruppage}"
ROOT="$PROJ/auto-research-benchmark"
DEV="$PROJ/memento-research-dev"

NAME="${1:?usage: run_sweep.sh <name> <tasks> <seeds> <max-boxes> <steps>}"
TASKS="${2:-all-lite}"
SEEDS="${3:-1}"
MAXBOXES="${4:-4}"
STEPS="${5:-8}"

export MLEBENCH_DATA_DIR="$PROJ/mlebench-data"
export MLEBENCH_PRIVATE_DATA_DIR="$PROJ/mlebench-data-private"  # grader-only split, firewalled from agents
export XDG_CACHE_HOME="$PROJ/.cache"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KEY="$(grep -E '^CUSTOM_API_KEY=' "$DEV/env-profiles/llm.api.env" | cut -d= -f2-)"

# env every remote box needs (LLM routing + data + fork repo for SHA stamping)
ENVX="export CUSTOM_API_KEY=$KEY; \
export OPENAI_BASE_URL=https://litellm.yangtzeailab.com/v1; export OPENAI_API_KEY=$KEY; \
export OPENAI_REQUEST_TIMEOUT=150 OPENAI_MAX_RETRIES=4; \
export ARBENCH_LLM_BACKEND=litellm ARBENCH_LLM_MODEL=Kimi-K2.6; \
export MLEBENCH_DATA_DIR=$PROJ/mlebench-data; \
export MLEBENCH_PRIVATE_DATA_DIR=$PROJ/mlebench-data-private; \
export XDG_CACHE_HOME=$PROJ/.cache; \
export KAGGLE_CONFIG_DIR=\$HOME/.kaggle; export AIDE_REPO=$ROOT/aideml;"

SWEEP="$ROOT/runs/$NAME"
mkdir -p "$SWEEP"
. "$ROOT/.venv/bin/activate"

echo "launching detached sweep -> $SWEEP (tasks=$TASKS seeds=$SEEDS max_boxes=$MAXBOXES steps=$STEPS)"
setsid arbench batch \
  --adapter aide --benchmark mlebench_lite \
  --tasks "$TASKS" --seeds "$SEEDS" --arms baseline \
  --out "$SWEEP" --max-boxes "$MAXBOXES" --steps "$STEPS" \
  --backend litellm --model Kimi-K2.6 \
  --data-dir "$PROJ/mlebench-data" \
  --venv "$ROOT/.venv/bin/activate" \
  --repo-dir "$ROOT" \
  --env-export "$ENVX" \
  > "$SWEEP/sweep.log" 2>&1 < /dev/null &

echo "PID=$!  log: $SWEEP/sweep.log  manifest: $SWEEP/manifest.json"
echo "watch:  tail -f $SWEEP/sweep.log"
