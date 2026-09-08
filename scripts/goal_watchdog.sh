#!/usr/bin/env bash
# Watch this node's goal run: report, and archive when it finishes.
#
# Installed on cron (*/10). It deliberately does NOT restart a dead run: that
# needs GOAL_TEACHER_KEY, and a key sitting in a cron line or a dotfile is
# readable by the teacher agent, which is the one thing this arm's design
# forbids. It records NEEDS_RESUME instead and the operator restarts.
#
# What it does do without the key: notice a stall, notice a full disk, and copy
# the finished checkpoints to the shared bucket -- 165 GB of an earlier study
# lived only on node-local disk for six days before anyone noticed.
set -u
cd "$(dirname "$0")/.." 2>/dev/null || exit 0
REPO="$PWD"
RUNS="$HOME/goal_runs"
OUT="$HOME/data/frontier-teacher/outputs/goal-teacher"
ARCH="$HOME/data/frontier-teacher/checkpoints"
STATUS="$RUNS/status.txt"
LOG="$RUNS/watchdog.log"
STALL_MIN=90          # a thinking step can legitimately take over an hour
mkdir -p "$RUNS"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >> "$LOG"; }

CFG=$(ls "$RUNS"/*.pid 2>/dev/null | head -1 | xargs -r basename | sed 's/\.pid$//')
[ -z "${CFG:-}" ] && { echo "no run on this node" > "$STATUS"; exit 0; }
PID=$(cat "$RUNS/$CFG.pid" 2>/dev/null || echo 0)

# $OUT is on the shared bucket and holds all three nodes' runs, so the newest one
# is usually somebody else's. Match on grpo_preset, which is the config name.
RUN=$(python3 - "$CFG" <<'PY'
import json, pathlib, sys
cfg = sys.argv[1]
out = pathlib.Path.home() / "data/frontier-teacher/outputs/goal-teacher"
for d in sorted(out.glob("run_*"), reverse=True):
    try:
        if json.loads((d / "pipeline" / "config.resolved.json").read_text()
                      )["student"]["grpo_preset"] == cfg:
            print(d)
            break
    except Exception:
        pass
PY
)
[ -z "${RUN:-}" ] && { echo "$CFG no-run-dir" > "$STATUS"; exit 0; }

alive=no; kill -0 "$PID" 2>/dev/null && alive=yes
done_=no; grep -q '"event": "run_done"' "$RUN/events.jsonl" 2>/dev/null && done_=yes
step=$(grep -o '"last_completed_step": [0-9-]*' "$RUN/run_state.json" 2>/dev/null \
       | tail -1 | grep -o '[0-9-]*$'); step=${step:--1}
age=$(( ( $(date +%s) - $(stat -c %Y "$RUN/events.jsonl" 2>/dev/null || echo 0) ) / 60 ))
free=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | paste -sd, - )

state=RUNNING
[ "$done_" = yes ] && state=COMPLETE
[ "$done_" = no ] && [ "$alive" = no ] && state=NEEDS_RESUME
[ "$state" = RUNNING ] && [ "$age" -gt "$STALL_MIN" ] && state=STALLED
[ -n "${free:-}" ] && [ "$free" -lt 20 ] && state="${state}+DISK_LOW"

printf '%s %s step=%s/20 alive=%s idle=%smin free=%sG run=%s\n' \
    "$CFG" "$state" "$step" "$alive" "$age" "${free:-?}" "$(basename "$RUN")" > "$STATUS"
say "$(cat "$STATUS") gpu=[$gpu]"

# Archive once, when it is over. rclone verifies; a partial copy is worse than
# none because it looks like a backup.
if [ "$done_" = yes ] && [ ! -f "$RUNS/$CFG.archived" ]; then
    src="$REPO/.local_checkpoints/$(basename "$RUN")"
    if [ -d "$src" ]; then
        say "archiving $src -> $ARCH/"
        mkdir -p "$ARCH/$(basename "$RUN")"
        if rclone copy "$src" "$ARCH/$(basename "$RUN")" --transfers 4 --checkers 8 \
                >> "$LOG" 2>&1 && \
           rclone check "$src" "$ARCH/$(basename "$RUN")" >> "$LOG" 2>&1; then
            touch "$RUNS/$CFG.archived"
            say "archive verified"
        else
            say "ARCHIVE FAILED -- checkpoints are still node-local only"
        fi
    fi
fi
