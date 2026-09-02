#!/usr/bin/env bash
# Evaluate a thinking run's checkpoints by sharding each job across ALL GPUs and
# running the jobs one at a time - the opposite of eval_checkpoints.sh, which
# gives each job its own GPU.
#
# For Llama and non-thinking, one-job-per-GPU is right: jobs are minutes long and
# the wave packs well. For thinking it fails twice over. A single HMMT job is
# 1,488 generations of up to 30,000 reasoning tokens and takes ~5.5 h on one
# card - longer than any timeout worth setting - while a wave of 8 runs only as
# fast as that slowest job, leaving cards idle for hours once the short ones
# finish. Measured on lumen-1: MATH-500 51 min, AIME 3.5 h, HMMT 5.5 h.
#
# Sharded, the same work is ~7 / 26 / 41 min per job. Identical GPU-hours,
# no job near a timeout, and no idle card.
#
#   eval_sharded.sh <config-name> <band-slug> [n_gpus] [tag-suffix]
set -u
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?}"; BAND="${2:?}"; NGPU="${3:-8}"; SUF="${4:-}"
EXP="${CFG}__${BAND}${SUF}"
# Checkpoints are node-local and must stay that way. outputs/ is now a symlink
# to the shared bucket, so writing 26 GB of weights per checkpoint there would
# push them over a fuse mount for no benefit: they are written, evaluated on the
# same node, and deleted, and no other node ever reads them.
CKROOT_BASE="$REPO/.local_checkpoints"
CKROOT="$CKROOT_BASE/$EXP"
TASKS="math500 aime hmmt"
mkdir -p logs outputs
[ -d "$CKROOT" ] || { echo "no checkpoints at $CKROOT"; exit 1; }

# Never launch onto a card that is still held. A reaped job can leave an
# EngineCore holding tens of GB, and the next process on that GPU dies at engine
# init with CUDA OOM - which is how shard 0 of one job was lost while its seven
# siblings finished normally. Killing the orphan is not enough; the driver
# releases asynchronously, so wait for the memory to actually drop.
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

for d in "$CKROOT"/global_step_*/actor/huggingface; do
  [ -f "$d/config.json" ] || continue
  step=$(echo "$d" | sed -E 's|.*/global_step_([0-9]+)/.*|\1|')
  for task in $TASKS; do
    tag="${EXP}__step${step}"
    summary="outputs/${tag}__${task}.summary.json"
    [ -f "$summary" ] && { echo "  skip (done) step $step / $task"; continue; }
    echo "  === step $step / $task across $NGPU GPUs  $(date -u +%H:%M:%S)Z ==="

    # Two rounds. A shard that dies at engine init - which happens when a
    # previous job left a process holding the card, and cost 2 h of waiting on
    # lumen-3 for a job whose other 7 shards had finished in 30 min - leaves no
    # summary, so the second round relaunches exactly the missing shards.
    for round in 1 2; do
      missing=""
      for g in $(seq 0 $((NGPU-1))); do
        [ -f "outputs/${tag}__${task}__s${g}of${NGPU}.summary.json" ] || missing="$missing $g"
      done
      [ -z "$missing" ] && break
      [ "$round" -eq 2 ] && echo "  retry shards:$missing"
      wait_gpus_free
      pids=()
      for g in $missing; do
        CUDA_VISIBLE_DEVICES=$g VLLM_LOGGING_LEVEL=WARNING \
          python eval/evaluate.py --config "configs/eval/${CFG}__${task}.yaml" \
            --weights "$d" --name "$tag" --shard "$g" --num-shards "$NGPU" \
            \
            > "logs/eval__${tag}__${task}__s${g}.log" 2>&1 &
        pids+=($!)
      done
      # vLLM regularly finishes its work and then fails to exit, so reap on the
      # shard's summary appearing rather than on the process ending. A shard is
      # ~30-50 min; an hour without one means it is stuck, not slow.
      want=$(echo $missing | wc -w); got=0; waited=0
      while [ "$got" -lt "$want" ] && [ "$waited" -lt 3600 ]; do
        sleep 10; waited=$((waited+10))
        got=0
        for g in $missing; do
          [ -f "outputs/${tag}__${task}__s${g}of${NGPU}.summary.json" ] && got=$((got+1))
        done
      done
      sleep 15
      for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
      for pid in $(ps -eo pid,cmd | grep '[V]LLM::EngineCore' | awk '{print $1}'); do
        kill -9 "$pid" 2>/dev/null
      done
      sleep 8
    done

    done_n=$(ls outputs/${tag}__${task}__s*of${NGPU}.summary.json 2>/dev/null | wc -l)
    if [ "$done_n" -eq "$NGPU" ]; then
      python eval/merge_shards.py --config "configs/eval/${CFG}__${task}.yaml" \
        --name "$tag" > "logs/merge__${tag}__${task}.log" 2>&1 \
        && rm -f outputs/${tag}__${task}__s*of${NGPU}.records.jsonl \
                 outputs/${tag}__${task}__s*of${NGPU}.summary.json \
        && echo "  ok   step $step / $task  $(date -u +%H:%M:%S)Z"
    else
      echo "  FAIL step $step / $task  ($done_n/$NGPU shards)"
    fi
  done
done
echo "=== $EXP: $(ls outputs/${EXP}__step*.summary.json 2>/dev/null | wc -l) summaries ==="
