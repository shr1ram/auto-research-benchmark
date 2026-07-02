#!/usr/bin/env bash
# Set up MLE-Bench (library + one Lite competition) on the UCL GPU box.
# Everything goes on the PROJECT filesystem — NFS home is tiny/full.
#
# Usage: scripts/setup_mlebench.sh <competition-id>
#   e.g. scripts/setup_mlebench.sh random-acts-of-pizza
#
# Prereqs (one-time, manual): put Kaggle creds at ~/.kaggle/kaggle.json and
# accept the competition's rules on kaggle.com, or `mlebench prepare` will 403.
set -euo pipefail

COMP="${1:-random-acts-of-pizza}"
PROJ="${PROJECT_FS:-/cs/student/project_msc/2025/csml/sruppage/thesis}"
ROOT="$PROJ/auto-research-benchmark"
MLE_SRC="$ROOT/mle-bench"
export MLEBENCH_DATA_DIR="${MLEBENCH_DATA_DIR:-$PROJ/mlebench-data}"
# Keep the appdirs cache off NFS home too.
export XDG_CACHE_HOME="$PROJ/.cache"

echo ">> project FS:        $PROJ"
echo ">> mlebench data dir: $MLEBENCH_DATA_DIR"
mkdir -p "$MLEBENCH_DATA_DIR"

# 1. clone + install mle-bench (idempotent)
if [ ! -d "$MLE_SRC/.git" ]; then
  git clone https://github.com/openai/mle-bench.git "$MLE_SRC"
fi
uv pip install -e "$MLE_SRC"

# 2. prepare one competition into the data dir
echo ">> preparing competition: $COMP"
mlebench prepare -c "$COMP" --data-dir "$MLEBENCH_DATA_DIR"

echo ">> done. Prepared data under $MLEBENCH_DATA_DIR/$COMP/prepared"
