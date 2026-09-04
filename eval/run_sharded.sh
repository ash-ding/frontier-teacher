#!/usr/bin/env bash
# Evaluate a run's checkpoints by sharding each job across ALL GPUs and running
# the jobs one at a time - the opposite of run_checkpoints.sh, which gives each
# job its own GPU.
#
# One job per GPU is right when jobs are minutes long. For thinking it fails
# twice over, measured: MATH-500 is 51 min on one card, AIME 3.5 h, HMMT 5.5 h,
# because a job is up to 1,488 generations of up to 30,000 reasoning tokens.
# That is longer than any timeout worth setting, and a wave of eight runs only
# as fast as its slowest member, leaving cards idle for hours. Sharded, the same
# work is ~7 / 26 / 41 min with nothing idle.
#
#   eval/run_sharded.sh <config-name> <band-slug> [n_gpus] [tag-suffix]
#
# CKROOT, OUTROOT and TASKS override where it looks, where it writes and what it
# runs -- which is how a teacher run is evaluated, its checkpoints being one
# directory deeper and all named global_step_1.
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?}"; BAND="${2:?}"; NGPU="${3:-8}"; SUF="${4:-}"
EXP="${CFG}__${BAND}${SUF}"
CKROOT="${CKROOT:-$REPO/.local_checkpoints/$EXP}"   # node-local; see run_checkpoints.sh
OUTROOT="${OUTROOT:-outputs/grpo}"
TASKS="${TASKS:-math500 aime hmmt}"
mkdir -p logs
[ -d "$CKROOT" ] || { echo "no checkpoints at $CKROOT"; exit 1; }

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

# Two layouts, as in run_checkpoints.sh: a GRPO run keeps global_step_<N>/ side
# by side; a teacher run gives each step its own step_<N>/ holding global_step_1.
for d in "$CKROOT"/global_step_*/actor/huggingface \
         "$CKROOT"/step_*/global_step_*/actor/huggingface; do
  [ -f "$d/config.json" ] || continue
  case "$d" in
    */step_*/global_step_*) step=$(echo "$d" | sed -E 's|.*/step_([0-9]+)/global_step_.*|\1|') ;;
    *)                      step=$(echo "$d" | sed -E 's|.*/global_step_([0-9]+)/.*|\1|') ;;
  esac
  for task in $TASKS; do
    od="${OUTROOT}/${EXP}__step${step}__${task}"
    [ -f "$od/summary.json" ] && { echo "  skip (done) step $step / $task"; continue; }
    echo "  === step $step / $task across $NGPU GPUs  $(date -u +%H:%M:%S)Z ==="

    # Two rounds. A shard that dies at engine init - which happens when a
    # previous job left a process holding the card - leaves no summary, and the
    # second round relaunches exactly the missing shards rather than all eight.
    for round in 1 2; do
      missing=""
      for g in $(seq 0 $((NGPU-1))); do
        [ -f "$od/summary.s${g}of${NGPU}.json" ] || missing="$missing $g"
      done
      [ -z "$missing" ] && break
      [ "$round" -eq 2 ] && echo "  retry shards:$missing"
      wait_gpus_free
      pids=()
      for g in $missing; do
        CUDA_VISIBLE_DEVICES=$g VLLM_LOGGING_LEVEL=WARNING \
          python eval/evaluate.py --config "configs/eval/${CFG}__${task}.yaml" \
            --model "$d" --output-path "$od" \
            --shard "$g" --num-shards "$NGPU" \
            > "logs/eval__${EXP}__step${step}__${task}__s${g}.log" 2>&1 &
        pids+=($!)
      done
      # vLLM regularly finishes its work and then fails to exit, so reap on the
      # shard's summary appearing. A shard is 30-50 min; an hour without one
      # means stuck, not slow.
      want=$(echo $missing | wc -w); got=0; waited=0
      while [ "$got" -lt "$want" ] && [ "$waited" -lt 3600 ]; do
        sleep 10; waited=$((waited+10))
        got=0
        for g in $missing; do
          [ -f "$od/summary.s${g}of${NGPU}.json" ] && got=$((got+1))
        done
      done
      sleep 15
      for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
      for pid in $(ps -eo pid,cmd | grep '[V]LLM::EngineCore' | awk '{print $1}'); do
        kill -9 "$pid" 2>/dev/null
      done
      sleep 8
    done

    n=$(ls "$od"/summary.s*of${NGPU}.json 2>/dev/null | wc -l)
    if [ "$n" -eq "$NGPU" ]; then
      python eval/merge_shards.py --output-path "$od" \
        > "logs/merge__${EXP}__step${step}__${task}.log" 2>&1 \
        && echo "  ok   step $step / $task  $(date -u +%H:%M:%S)Z"
    else
      echo "  FAIL step $step / $task  ($n/$NGPU shards)"
    fi
  done
done
echo "=== $EXP: $(ls -d ${OUTROOT}/${EXP}__step*/summary.json 2>/dev/null | wc -l) evaluations complete ==="
