#!/usr/bin/env bash
# Wait for a teacher run to finish, then evaluate its four kept checkpoints on
# all three benchmarks in full.
#
#   eval/after_teacher_run.sh <config-name> <run-dir> [n_gpus]
#
# The run's own per-step reference tests are 100 / 30 / 19 problems -- enough to
# watch, too few to conclude from. This is the full 500 / 150 / 93, which is what
# puts a teacher curve on the same footing as the baseline bands in outputs/grpo/.
#
# It waits on the orchestrator by PID, never by name pattern: `pkill -f` and
# `pgrep -f` have killed the ssh session running them twice in this project.
# A run that halted short of its last step is evaluated anyway, on whatever
# checkpoints it kept -- with a line saying so, because a curve that stops early
# does not have the endpoint the others report.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"

CFG="${1:?usage: eval/after_teacher_run.sh <config-name> <run-dir> [n_gpus]}"
RUN="${2:?missing <run-dir>}"
NGPU="${3:-8}"
RUN_NAME="$(basename "$RUN")"

pid=$(ps -eo pid,args --no-headers |
        awk -v c="orchestrator.py --config configs/grpo/$CFG" \
            'index($0, c) { print $1; exit }')
if [ -n "$pid" ]; then
  echo "$(date +%H:%M:%S) waiting for orchestrator pid $pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
fi

steps=$(ls -d "$RUN"/step_*/result.json 2>/dev/null | wc -l)
echo "$(date +%H:%M:%S) run finished with $steps closed step(s)"
[ "$steps" -lt 20 ] && echo "  NOTE: fewer than 20 steps -- this curve stops short of 10,240 rollouts"

CKROOT="$REPO/.local_checkpoints/$RUN_NAME" \
OUTROOT="$RUN/final_eval" \
  bash eval/run_checkpoints.sh "$CFG" teacher "$NGPU"
