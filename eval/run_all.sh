#!/usr/bin/env bash
# Baseline evaluation: every (config, task) pair over the three benchmarks,
# one GPU each, with raw generations saved.
#
# Generations were NOT saved on the first pass, which is why the original six
# baseline runs have no reasoning traces - the data does not exist anywhere.
# Generations are written unconditionally by eval/evaluate.py; there is no
# flag left to forget.
set -u
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CONFIGS="${CONFIGS:-llama32-3b qwen3-4b-think qwen3-4b-nothink}"
TASKS="${TASKS:-math500 aime hmmt}"
mkdir -p logs outputs

i=0
for cfg in $CONFIGS; do
  for task in $TASKS; do
    log="logs/${cfg}__${task}.log"
    CUDA_VISIBLE_DEVICES=$((i % 8)) VLLM_LOGGING_LEVEL=WARNING \
      nohup python eval/evaluate.py --config "configs/eval/${cfg}__${task}.yaml" \
        --output-path "outputs/benchmarks/${cfg}__${task}" \
        > "$log" 2>&1 &
    echo "  gpu $((i % 8))  ${cfg}/${task}  pid $!"
    i=$((i+1))
  done
done
echo "launched $i jobs; follow with: tail -f logs/*.log"
