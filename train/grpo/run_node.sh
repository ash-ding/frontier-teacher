#!/usr/bin/env bash
# Everything one node does for one model configuration: three GRPO runs, each
# followed immediately by evaluation of its checkpoints, then deletion of those
# checkpoints so the next run has disk.
#
#   train/grpo/run_node.sh <config-name> [band-slugs...]
#
# Evaluating in-place, per run, rather than batching all evaluation to the end,
# means a run that finishes is fully reported even if a later one fails.
set -u
cd "$(dirname "$0")/../.."
CFG="${1:?usage: run_node.sh <config-name> [bands...]}"
shift || true
case "$CFG" in
  llama32-3b)       DEFAULT_BANDS="pass1_05-15pct pass1_40-60pct pass1_85-95pct" ;;
  qwen3-4b-nothink|qwen3-4b-think) DEFAULT_BANDS="pass1_12-25pct pass1_37-62pct pass1_75-87pct" ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac
BANDS="${*:-$DEFAULT_BANDS}"
mkdir -p logs

for band in $BANDS; do
  EXP="${CFG}__${band}"
  echo "############ $(date +%H:%M:%S)  TRAIN $EXP ############"
  ./train/grpo/run_grpo.sh "$CFG" "$band" 8 > "logs/grpo__${EXP}.log" 2>&1
  rc=$?
  echo "  train exit=$rc  checkpoints=$(ls -d .local_checkpoints/$EXP/global_step_* 2>/dev/null | wc -l)"

  echo "############ $(date +%H:%M:%S)  EVAL $EXP ############"
  ./eval/run_checkpoints.sh "$CFG" "$band" 8 >> "logs/eval__${EXP}.log" 2>&1
  echo "  summaries=$(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l)"

  # keep only the HF weights we may want later; drop the FSDP shards
  find ".local_checkpoints/$EXP" -name 'model_world_size_*' -delete 2>/dev/null
  echo "  disk after cleanup: $(df -h . | tail -1 | awk '{print $4}') free"
done
echo "############ $(date +%H:%M:%S)  NODE DONE: $CFG ############"
