#!/usr/bin/env bash
# Evaluate every checkpoint of one GRPO run on all three benchmarks, using the
# unchanged evaluation harness so the numbers join the existing baseline table.
#
#   scripts/eval_checkpoints.sh <config-name> <band-slug> [n_gpus]
#
# --name is what keeps this safe: cfg["name"] forms the output tag, so without an
# override every checkpoint would overwrite the model's baseline summary.
#
# vLLM processes here regularly finish their work - summary written, GPU released -
# and then fail to exit. A plain `wait` therefore blocks forever and starves the
# rest of the queue. Each job is instead reaped once its summary appears, and
# capped by a hard timeout as a backstop.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: eval_checkpoints.sh <config-name> <band-slug> [n_gpus]}"
BAND="${2:?}"
NGPU="${3:-8}"
JOB_TIMEOUT="${JOB_TIMEOUT:-3600}"
EXP="${CFG}__${BAND}"
CKROOT="$REPO/outputs/checkpoints/$EXP"
TASKS="math500 aime hmmt"
mkdir -p logs outputs
[ -d "$CKROOT" ] || { echo "no checkpoints at $CKROOT"; exit 1; }

run_one () {   # gpu step task dir
  local gpu=$1 step=$2 task=$3 dir=$4
  local tag="${EXP}__step${step}"
  local summary="outputs/${tag}__${task}.summary.json"
  [ -f "$summary" ] && { echo "  skip (done) step $step / $task"; return 0; }

  CUDA_VISIBLE_DEVICES=$gpu VLLM_LOGGING_LEVEL=WARNING \
    python src/evaluate.py --config "configs/${CFG}.yaml" --task "$task" \
      --model "$dir" --name "$tag" > "logs/eval__${tag}__${task}.log" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ -f "$summary" ]; then
      sleep 10                       # let it flush records/generations
      kill -9 "$pid" 2>/dev/null      # then reap; it will not exit on its own
      break
    fi
    [ "$waited" -ge "$JOB_TIMEOUT" ] && { kill -9 "$pid" 2>/dev/null; echo "  TIMEOUT step $step / $task"; break; }
    sleep 5; waited=$((waited+5))
  done
  wait "$pid" 2>/dev/null
  [ -f "$summary" ] && echo "  ok   step $step / $task" || echo "  FAIL step $step / $task"
}

JOBS=()
for d in "$CKROOT"/global_step_*/actor/huggingface; do
  [ -f "$d/config.json" ] || continue
  step=$(echo "$d" | sed -E 's|.*/global_step_([0-9]+)/.*|\1|')
  for t in $TASKS; do JOBS+=("$step $t $d"); done
done
echo "=== $EXP: ${#JOBS[@]} eval jobs across $NGPU GPUs ==="

i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  pids=()
  for g in $(seq 0 $((NGPU-1))); do
    [ "$i" -ge "${#JOBS[@]}" ] && break
    set -- ${JOBS[$i]}
    run_one "$g" "$1" "$2" "$3" &
    pids+=($!)
    i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done   # these are the reapers, not vLLM
  # clear any EngineCore orphaned by the reaped processes
  for pid in $(ps -eo pid,cmd | grep '[V]LLM::EngineCore' | awk '{print $1}'); do
    kill -9 "$pid" 2>/dev/null
  done
done
echo "=== $EXP evaluation complete: $(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l) summaries ==="
