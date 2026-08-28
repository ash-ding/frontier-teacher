#!/usr/bin/env bash
# Evaluate every checkpoint of one GRPO run on all three benchmarks, using the
# unchanged evaluation harness so the numbers join the existing baseline table.
#
#   scripts/eval_checkpoints.sh <config-name> <band-slug> [n_gpus]
#
# --name is what keeps this safe: cfg["name"] forms the output tag, so without an
# override every checkpoint would overwrite the model's baseline summary.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: eval_checkpoints.sh <config-name> <band-slug> [n_gpus]}"
BAND="${2:?}"
NGPU="${3:-8}"
EXP="${CFG}__${BAND}"
CKROOT="$REPO/outputs/checkpoints/$EXP"
TASKS="math500 aime hmmt"
mkdir -p logs outputs

[ -d "$CKROOT" ] || { echo "no checkpoints at $CKROOT"; exit 1; }

# build the job queue: one (checkpoint, task) pair per line
QUEUE=$(mktemp)
for d in "$CKROOT"/global_step_*/actor/huggingface; do
  [ -f "$d/config.json" ] || continue
  step=$(echo "$d" | sed -E 's|.*/global_step_([0-9]+)/.*|\1|')
  for t in $TASKS; do echo "$step $t $d"; done
done > "$QUEUE"
echo "=== $EXP: $(wc -l < "$QUEUE") eval jobs across $NGPU GPUs ==="

g=0
while read -r step task dir; do
  tag="${EXP}__step${step}"
  CUDA_VISIBLE_DEVICES=$g VLLM_LOGGING_LEVEL=WARNING \
    nohup python src/evaluate.py --config "configs/${CFG}.yaml" --task "$task" \
      --model "$dir" --name "$tag" > "logs/eval__${tag}__${task}.log" 2>&1 &
  echo "  gpu $g  step $step / $task"
  g=$((g+1))
  if [ "$g" -ge "$NGPU" ]; then wait; g=0; fi
done < "$QUEUE"
wait
rm -f "$QUEUE"
echo "=== $EXP evaluation complete: $(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l) summaries ==="
