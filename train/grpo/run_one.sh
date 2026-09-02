#!/usr/bin/env bash
# One band, train then evaluate. Used to spread a slow configuration's bands
# across nodes as they free up, instead of running them serially on one node.
#   train/grpo/run_one.sh <config-name> <band-slug>
set -u
cd "$(dirname "$0")/../.."
CFG="${1:?}"; BAND="${2:?}"
# TAG suffixes the experiment name so a re-run under different settings
# (see GROUP_SIZE in run_grpo.sh) writes beside the original instead of
# overwriting it. Both are passed through to the scripts below.
export TAG="${TAG:-}" GROUP_SIZE="${GROUP_SIZE:-}"
EXP="${CFG}__${BAND}${TAG:-}"
mkdir -p logs
echo "############ $(date +%H:%M:%S)  TRAIN $EXP ############"
./train/grpo/run_grpo.sh "$CFG" "$BAND" 8 > "logs/grpo__${EXP}.log" 2>&1
echo "  train exit=$?  checkpoints=$(ls -d outputs/checkpoints/$EXP/global_step_* 2>/dev/null | wc -l)"
echo "############ $(date +%H:%M:%S)  EVAL $EXP ############"
./eval/run_checkpoints.sh "$CFG" "$BAND" 8 >> "logs/eval__${EXP}.log" 2>&1
echo "  summaries=$(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l)"
find "outputs/checkpoints/$EXP" -name 'model_world_size_*' -delete 2>/dev/null
echo "############ $(date +%H:%M:%S)  ONE DONE: $EXP ############"
