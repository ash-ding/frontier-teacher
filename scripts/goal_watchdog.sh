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

# Offload finished milestones as they appear, rather than only at the end.
#
# A step reads exactly one checkpoint -- step_{N-1}'s HF export -- so the
# milestones kept at 4, 9 and 14 are never opened again by the run that wrote
# them; they are kept for the evaluation afterwards. Holding all four locally
# plus the one being written peaks at five checkpoints, which on this node is
# 72 GB against 80 GB of pool that is also shared with other containers.
#
# So each milestone below the current step is copied to the bucket, verified,
# and only then removed locally. That is where the end-of-run archive would put
# it anyway; this just does it sooner. Nothing is deleted that has not been
# checked byte for byte first, and a failed check leaves the local copy alone.
if [ "$step" -ge 0 ]; then
    for d in "$REPO/.local_checkpoints/$(basename "$RUN")"/step_*; do
        [ -d "$d" ] || continue
        n=$(basename "$d" | tr -dc '0-9')
        [ -z "$n" ] && continue
        [ "$n" -ge "$step" ] && continue                 # never touch the live one
        [ $(( (n + 1) % 5 )) -ne 0 ] && continue          # only milestones
        dst="$ARCH/$(basename "$RUN")/step_$n"
        mkdir -p "$dst"
        rclone copy "$d" "$dst" --transfers 4 --checkers 8 >> "$LOG" 2>&1
        # The bucket is a write-back FUSE mount: a check fired the instant copy
        # returns sees files the cache has not flushed and reports a difference
        # that is not one. Retry before believing it -- the whole point of the
        # check is that a false negative here costs a re-copy while a false
        # positive deletes the only local copy.
        ok=no
        for try in 1 2 3; do
            if rclone check "$d" "$dst" >> "$LOG" 2>&1; then ok=yes; break; fi
            sleep 30
        done
        if [ "$ok" = yes ]; then
            rm -rf "$d"
            say "offloaded step_$n to the bucket and freed it locally"
        else
            say "step_$n offload FAILED after 3 checks; local copy kept"
        fi
    done
fi

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
