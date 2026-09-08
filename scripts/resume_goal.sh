#!/usr/bin/env bash
# Resume this node's goal run from wherever it stopped.
#
#   GOAL_TEACHER_KEY=... scripts/resume_goal.sh <config-name>
#
# The loop halts cleanly on the first protocol failure rather than retrying, so a
# halt is normal operation and this is how it is answered. Position is rebuilt
# from the artifacts on disk, not from anything passed here: a step counts as
# closed only when its result.json validates and names a checkpoint, so a step
# that died mid-train is simply redone.
set -u
cd "$(dirname "$0")/.."
CFG="${1:?usage: resume_goal.sh <config-name>}"
: "${GOAL_TEACHER_KEY:?set GOAL_TEACHER_KEY}"

# The output directory is shared by all three nodes, so "the newest run" is not
# this node's run. Match on grpo_preset, which is the config name.
RUN=$(python3 - "$CFG" <<'PY'
import json, pathlib, sys
cfg = sys.argv[1]
out = pathlib.Path.home() / "data/frontier-teacher/outputs/goal-teacher"
best = None
for d in sorted(out.glob("run_*"), reverse=True):
    p = d / "pipeline" / "config.resolved.json"
    try:
        if json.loads(p.read_text())["student"]["grpo_preset"] == cfg:
            best = d
            break
    except Exception:
        pass
print(best or "")
PY
)
[ -z "${RUN:-}" ] && { echo "no run dir for $CFG"; exit 1; }

if grep -q '"event": "run_done"' "$RUN/events.jsonl" 2>/dev/null; then
    echo "$(basename "$RUN") already finished; nothing to resume"
    exit 0
fi
PID=$(cat "$HOME/goal_runs/$CFG.pid" 2>/dev/null || echo 0)
if kill -0 "$PID" 2>/dev/null; then
    echo "pid $PID is still alive; refusing to start a second writer"
    exit 1
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate frontier-teacher
LOG="$HOME/goal_runs/${CFG}_resume_$(date +%Y%m%d_%H%M%S).log"

setsid nohup python3 train/goal_teacher/orchestrator.py \
    --config "configs/goal/${CFG}-goal.yaml" --resume "$RUN" \
    > "$LOG" 2>&1 < /dev/null &
NEW=$!
echo "$NEW" > "$HOME/goal_runs/${CFG}.pid"
sleep 5
kill -0 "$NEW" 2>/dev/null \
    && echo "resumed ${CFG} from $(basename "$RUN")  pid=${NEW}  log=${LOG}" \
    || { echo "DIED IMMEDIATELY"; tail -20 "$LOG"; exit 1; }
