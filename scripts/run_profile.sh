#!/usr/bin/env bash
# Difficulty-profile one model over the 12k MATH train pool, sharded across 8 GPUs.
# usage: ./scripts/run_profile.sh <config-name>
set -u
cd "$(dirname "$0")/.."
CFG="${1:?usage: run_profile.sh <config-name>}"
TASK=mathtrain
GENDIR="$HOME/data/frontier-teacher/generations"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher
mkdir -p logs outputs "$GENDIR"

N=8
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i VLLM_LOGGING_LEVEL=WARNING \
    nohup python src/evaluate.py --config "configs/${CFG}.yaml" --task "$TASK" \
      --shard "$i" --num-shards "$N" --save-generations "$GENDIR" \
      > "logs/${CFG}__${TASK}__s${i}.log" 2>&1 &
  echo "  gpu $i  shard $i/$N  pid $!"
done
echo "launched $N shards for ${CFG}/${TASK}"
echo "merge when done:  python src/merge_shards.py --config configs/${CFG}.yaml --task ${TASK}"
