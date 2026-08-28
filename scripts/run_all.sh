#!/usr/bin/env bash
# Launch all 6 (config, task) jobs concurrently, one GPU each.
set -u
cd "$(dirname "$0")/.."
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher
mkdir -p logs outputs

i=0
for cfg in llama32-3b qwen3-4b-think qwen3-4b-nothink; do
  for task in math500 aime; do
    log="logs/${cfg}__${task}.log"
    CUDA_VISIBLE_DEVICES=$i VLLM_LOGGING_LEVEL=WARNING \
      nohup python src/evaluate.py --config "configs/${cfg}.yaml" --task "$task" \
      > "$log" 2>&1 &
    echo "gpu $i  ${cfg}/${task}  pid $!  -> $log"
    i=$((i+1))
  done
done
echo "launched $i jobs; wait with: tail -f logs/*.log"
