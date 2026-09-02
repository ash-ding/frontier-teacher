#!/usr/bin/env bash
# Evaluate every checkpoint of one GRPO run on all three benchmarks, one job per
# GPU. Right for Llama and non-thinking, where a job is minutes; use
# run_sharded.sh for thinking, where one HMMT job is 5.5 h on a single card.
#
#   eval/run_checkpoints.sh <config-name> <band-slug> [n_gpus]
#
# Each job gets its own output directory, which is what keeps a checkpoint's
# score from landing on top of the baseline it is compared against.
#
# vLLM processes here regularly finish their work - summary written, GPU
# released - and then fail to exit. A plain `wait` therefore blocks forever and
# starves the queue. Each job is reaped once its summary appears, with a hard
# timeout as a backstop.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: eval/run_checkpoints.sh <config-name> <band-slug> [n_gpus]}"
BAND="${2:?}"
NGPU="${3:-8}"
case "$CFG" in
  *think*) JOB_TIMEOUT="${JOB_TIMEOUT:-14400}" ;;
  *)       JOB_TIMEOUT="${JOB_TIMEOUT:-3600}"  ;;
esac
EXP="${CFG}__${BAND}${TAG:-}"
# Checkpoints are node-local: outputs/ is a symlink to the shared bucket, and
# 26 GB of weights per checkpoint has no business crossing a fuse mount.
CKROOT="$REPO/.local_checkpoints/$EXP"
TASKS="math500 aime hmmt"
mkdir -p logs
[ -d "$CKROOT" ] || { echo "no checkpoints at $CKROOT"; exit 1; }

# Never launch onto a card still held. A reaped job can leave an EngineCore
# holding tens of GB, and the next process there dies at engine init with CUDA
# OOM. Killing the orphan is not enough: the driver releases asynchronously.
wait_gpus_free () {
  local tries=0 busy
  while [ "$tries" -lt 60 ]; do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
             awk '$1 > 2000' | wc -l)
    [ "$busy" -eq 0 ] && return 0
    [ "$tries" -eq 0 ] && echo "  waiting for $busy GPU(s) to drain"
    sleep 10; tries=$((tries + 1))
  done
  echo "  WARNING: GPUs still busy after 10 min, launching anyway"
}

run_job () {   # gpu step task ckpt-dir
  local gpu=$1 step=$2 task=$3 dir=$4
  local od="outputs/grpo/${EXP}__step${step}__${task}"
  [ -f "$od/summary.json" ] && { echo "  skip (done) step $step / $task"; return 0; }

  CUDA_VISIBLE_DEVICES=$gpu VLLM_LOGGING_LEVEL=WARNING \
    python eval/evaluate.py --config "configs/eval/${CFG}__${task}.yaml" \
      --model "$dir" --output-path "$od" \
      > "logs/eval__${EXP}__step${step}__${task}.log" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ -f "$od/summary.json" ]; then
      sleep 10; kill -9 "$pid" 2>/dev/null; break
    fi
    [ "$waited" -ge "$JOB_TIMEOUT" ] && { kill -9 "$pid" 2>/dev/null; echo "  TIMEOUT step $step / $task"; break; }
    sleep 5; waited=$((waited+5))
  done
  wait "$pid" 2>/dev/null
  [ -f "$od/summary.json" ] && echo "  ok   step $step / $task" || echo "  FAIL step $step / $task"
}

JOBS=()
for d in "$CKROOT"/global_step_*/actor/huggingface; do
  [ -f "$d/config.json" ] || continue
  step=$(echo "$d" | sed -E 's|.*/global_step_([0-9]+)/.*|\1|')
  for t in $TASKS; do JOBS+=("$step $t $d"); done
done
echo "=== $EXP: ${#JOBS[@]} eval jobs across $NGPU GPUs ==="

done_count () { ls -d outputs/grpo/${EXP}__step*/summary.json 2>/dev/null | wc -l; }

# Two passes: a job that failed on a busy GPU leaves no summary, and the second
# pass picks it up while finished jobs are skipped.
for attempt in 1 2; do
  [ "$attempt" -eq 2 ] && { [ "$(done_count)" -ge "${#JOBS[@]}" ] && break
                            echo "=== retry pass ==="; }
  i=0
  while [ "$i" -lt "${#JOBS[@]}" ]; do
    wait_gpus_free
    pids=()
    for g in $(seq 0 $((NGPU-1))); do
      [ "$i" -ge "${#JOBS[@]}" ] && break
      set -- ${JOBS[$i]}
      run_job "$g" "$1" "$2" "$3" &
      pids+=($!)
      i=$((i+1))
    done
    for p in "${pids[@]}"; do wait "$p"; done   # the reapers, not vLLM
    for pid in $(ps -eo pid,cmd | grep '[V]LLM::EngineCore' | awk '{print $1}'); do
      kill -9 "$pid" 2>/dev/null
    done
  done
done
echo "=== $EXP: $(done_count)/${#JOBS[@]} evaluations complete ==="
