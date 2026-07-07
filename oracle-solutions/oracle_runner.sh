#!/usr/bin/env bash
# Oracle harness: run a solution for a task, time it, GPU-sample it, grade it, save everything.
# Usage: oracle_runner.sh <task_id> <solution.py path>
set -u
TASK="$1"; SOL="$2"
PROJ=/cs/student/project_msc/2025/csml/sruppage
ROOT=$PROJ/auto-research-benchmark
export MLEBENCH_DATA_DIR=$PROJ/mlebench-data XDG_CACHE_HOME=$PROJ/.cache
. "$ROOT/.venv/bin/activate"
OD="$ROOT/oracle-solutions/$TASK"
mkdir -p "$OD"
cp "$SOL" "$OD/solution.py"
DATA="$MLEBENCH_DATA_DIR/$TASK/prepared/public"
cd "$OD"
# GPU sampler
( while true; do echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"; sleep 5; done ) > "$OD/gpu_samples.csv" 2>/dev/null &
SAMP=$!
S=$(date +%s)
# run the solution with DATA_DIR env; it must write submission.csv in CWD
DATA_DIR="$DATA" timeout 1800 python "$OD/solution.py" > "$OD/run.log" 2>&1
RC=$?
E=$(date +%s); WALL=$((E-S))
kill $SAMP 2>/dev/null
# peak gpu
PEAK=$(awk -F, 'NR>0{if($2+0>m)m=$2} END{print m+0}' "$OD/gpu_samples.csv" 2>/dev/null)
MAXU=$(awk -F, 'NR>0{if($3+0>m)m=$3} END{print m+0}' "$OD/gpu_samples.csv" 2>/dev/null)
# grade if submission exists
SCORE="null"; VALID="false"; MEDIAN="null"; GOLD="null"; ABOVE="null"
if [ -f "$OD/submission.csv" ]; then
python - "$TASK" "$OD/submission.csv" "$OD" <<PY
import json,sys,os
os.environ.setdefault("MLEBENCH_DATA_DIR","$MLEBENCH_DATA_DIR")
from arbench.benchmarks.mlebench_lite.benchmark import MLEBenchLite
b=MLEBenchLite(); t=b.load_task(sys.argv[1])
try:
    sc=b.grade(t, sys.argv[2]); d=sc.details or {}
    out=dict(score=sc.value, valid=sc.valid, median=d.get("median_threshold"), gold=d.get("gold_threshold"), above_median=d.get("above_median"), any_medal=d.get("any_medal"))
except Exception as e:
    out=dict(score=None, valid=False, grade_error=str(e))
json.dump(out, open(sys.argv[3]+"/_grade.json","w"))
print("GRADED", out.get("score"))
PY
fi
# assemble result.json
python - "$TASK" "$WALL" "$RC" "$PEAK" "$MAXU" "$OD" <<PY
import json,sys,os
task,wall,rc,peak,maxu,od=sys.argv[1:7]
g={}
gp=od+"/_grade.json"
if os.path.exists(gp): g=json.load(open(gp))
sub_exists=os.path.exists(od+"/submission.csv")
res=dict(task=task, ran=(rc=="0"), exit_code=int(rc), wall_clock_s=int(wall),
         submission_written=sub_exists, peak_gpu_mib=int(peak), max_gpu_util=int(maxu),
         score=g.get("score"), valid=g.get("valid"), median=g.get("median"), gold=g.get("gold"),
         above_median=g.get("above_median"), any_medal=g.get("any_medal"), grade_error=g.get("grade_error"))
json.dump(res, open(od+"/result.json","w"), indent=2)
print("ORACLE_RESULT", json.dumps(res))
PY
