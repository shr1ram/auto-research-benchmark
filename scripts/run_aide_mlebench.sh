#!/usr/bin/env bash
# Run AIDE on one MLE-Bench Lite competition through the arbench harness.
#
# Usage: scripts/run_aide_mlebench.sh <competition-id> [steps]
#
# Assumes: arbench + AIDE + mlebench installed, the competition prepared
# (scripts/setup_mlebench.sh), and the LiteLLM key in env (CUSTOM_API_KEY).
set -euo pipefail

COMP="${1:-random-acts-of-pizza}"
STEPS="${2:-10}"
PROJ="${PROJECT_FS:-/cs/student/project_msc/2025/csml/sruppage}"
export MLEBENCH_DATA_DIR="${MLEBENCH_DATA_DIR:-$PROJ/mlebench-data}"
export MLEBENCH_PRIVATE_DATA_DIR="${MLEBENCH_PRIVATE_DATA_DIR:-$PROJ/mlebench-data-private}"  # grader-only split
export XDG_CACHE_HOME="$PROJ/.cache"

WS="$PROJ/auto-research-benchmark/runs/${COMP}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$WS"

echo ">> running AIDE on $COMP ($STEPS steps), backend=litellm (Kimi-K2.6)"
arbench run \
  --adapter aide \
  --benchmark mlebench_lite \
  --task "$COMP" \
  --backend litellm \
  --steps "$STEPS" \
  --data-dir "$MLEBENCH_DATA_DIR" \
  --workspace "$WS" \
  --out "$WS/result.json"

echo ">> result:"
cat "$WS/result.json"
