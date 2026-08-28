#!/usr/bin/env bash
# One band, train then evaluate. Used to spread a slow configuration's bands
# across nodes as they free up, instead of running them serially on one node.
#   scripts/run_one.sh <config-name> <band-slug>
set -u
cd "$(dirname "$0")/.."
CFG="${1:?}"; BAND="${2:?}"
EXP="${CFG}__${BAND}"
mkdir -p logs
echo "############ $(date +%H:%M:%S)  TRAIN $EXP ############"
./scripts/run_grpo.sh "$CFG" "$BAND" 8 > "logs/grpo__${EXP}.log" 2>&1
echo "  train exit=$?  checkpoints=$(ls -d outputs/checkpoints/$EXP/global_step_* 2>/dev/null | wc -l)"
echo "############ $(date +%H:%M:%S)  EVAL $EXP ############"
./scripts/eval_checkpoints.sh "$CFG" "$BAND" 8 >> "logs/eval__${EXP}.log" 2>&1
echo "  summaries=$(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l)"
find "outputs/checkpoints/$EXP" -name 'model_world_size_*' -delete 2>/dev/null
echo "############ $(date +%H:%M:%S)  ONE DONE: $EXP ############"
