#!/usr/bin/env bash
# Launch one goal-directed run on this node, detached, and print where it went.
#
#   GOAL_TEACHER_KEY=... launch_goal.sh <config-name>
#
# Detached with setsid+nohup rather than a bare & : the ssh session that starts
# it will not survive the night, and a run that dies with its shell has burned
# its teacher spend for nothing.
set -u
cd "$(dirname "$0")/.."
CFG="${1:?usage: launch_goal.sh <config-name, e.g. llama32-3b>}"
: "${GOAL_TEACHER_KEY:?set GOAL_TEACHER_KEY}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate frontier-teacher

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$HOME/goal_runs/${CFG}_${STAMP}.log"
mkdir -p "$HOME/goal_runs"

setsid nohup python3 train/goal_teacher/orchestrator.py \
    --config "configs/goal/${CFG}-goal.yaml" \
    > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$HOME/goal_runs/${CFG}.pid"
sleep 5
if kill -0 "$PID" 2>/dev/null; then
    echo "started ${CFG}  pid=${PID}  log=${LOG}"
else
    echo "DIED IMMEDIATELY -- log follows"
    tail -20 "$LOG"
    exit 1
fi
