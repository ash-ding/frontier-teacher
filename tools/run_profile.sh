#!/usr/bin/env bash
# Difficulty-profile one model over the 12k MATH train pool, sharded across 8 GPUs.
# Output lands in outputs/math_profiling/ - the mathtrain task declares that via
# out_subdir, so it is not something this script can forget to pass.
# usage: ./scripts/run_profile.sh <config-name>
set -u
cd "$(dirname "$0")/.."
CFG="${1:?usage: run_profile.sh <config-name>}"
TASK=mathtrain

source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher
mkdir -p logs outputs

N=8
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i VLLM_LOGGING_LEVEL=WARNING \
    nohup python eval/evaluate.py --config "configs/eval/${CFG}.yaml" --task "$TASK" \
      --shard "$i" --num-shards "$N" \
      > "logs/${CFG}__${TASK}__s${i}.log" 2>&1 &
  echo "  gpu $i  shard $i/$N  pid $!"
done
echo "launched $N shards for ${CFG}/${TASK}"
echo "merge when done:  python eval/merge_shards.py --config configs/eval/${CFG}.yaml --task ${TASK}"
