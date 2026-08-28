#!/usr/bin/env bash
# One GRPO run: scripts/run_grpo.sh <config-name> <band-slug> [n_gpus]
#
#   scripts/run_grpo.sh llama32-3b pass1_05-15pct
#
# Band -> group size follows the measured pass@1 of each band. A GRPO group whose
# G rollouts are all wrong contributes zero gradient, and the wasted fraction is
# (1-p)^G: at p=8.7% that is 48% for G=8 but 5% for G=32, while at p=50% even G=8
# wastes under 1%. So the extreme bands get G=32 and the middle band G=8.
#
# Batch sizes are chosen so every run consumes 512 rollouts per step, which puts
# all nine curves on a common x-axis.
#
# trainer.use_v1=False: verl 0.9.0's v1 trainer imports `transfer_queue`, which is
# neither on PyPI nor declared as a verl dependency - a packaging gap in the
# release. The v0 path (verl.trainer.ppo.ray_trainer) imports cleanly and reads
# the same reward.* config keys.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: run_grpo.sh <config-name> <band-slug> [n_gpus]}"
BAND="${2:?usage: run_grpo.sh <config-name> <band-slug> [n_gpus]}"
NGPU="${3:-8}"

case "$CFG" in
  llama32-3b)       MODEL=unsloth/Llama-3.2-3B-Instruct; MAXRESP=4096;  TEMP=0.6; TOPP=0.9;  TOPK=-1; THINK='' ;;
  qwen3-4b-nothink) MODEL=Qwen/Qwen3-4B;                 MAXRESP=8192;  TEMP=0.7; TOPP=0.8;  TOPK=20; THINK=false ;;
  qwen3-4b-think)   MODEL=Qwen/Qwen3-4B;                 MAXRESP=12288; TEMP=0.6; TOPP=0.95; TOPK=20; THINK=true ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

# middle band -> G=8; the two extreme bands -> G=32. Both give 512 rollouts/step.
case "$BAND" in
  *37-62pct*|*40-60pct*) G=8;  TB=64; MB=32 ;;
  *)                     G=32; TB=16; MB=8  ;;
esac

SUB=$(ls data/further_improve/"$CFG"/"$CFG"__"$BAND"__n*.jsonl 2>/dev/null | head -1)
[ -n "$SUB" ] || { echo "no subset matching $CFG/$BAND"; exit 1; }
TRAIN="${SUB%.jsonl}.verl.jsonl"
[ -f "$TRAIN" ] || python src/to_verl_dataset.py --subset "$SUB"

EXP="${CFG}__${BAND}"
# Checkpoints go to the LOCAL container disk, not ~/data. One checkpoint is 26 GB
# with optimizer state dropped, so the 36 across all runs would be ~940 GB of
# writes over the rclone bucket mount. The orchestrator evaluates each run's
# checkpoints on the same node and deletes them before the next run starts;
# per-run peak is 4 x 26 GB = 104 GB against 174 GB free.
CKPT="$REPO/outputs/checkpoints/$EXP"
mkdir -p "$CKPT" logs

echo "=== $EXP ==="
echo "  model=$MODEL  subset=$(basename "$SUB")  problems=$(wc -l < "$SUB")"
echo "  G=$G  train_batch=$TB  mini_batch=$MB  -> $((TB*G)) rollouts/step, 20 steps = $((TB*G*20))"
echo "  max_response=$MAXRESP  enable_thinking=${THINK:-n/a}  gpus=$NGPU"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$REPO/$TRAIN" \
  data.val_files="$REPO/$TRAIN" \
  data.train_batch_size=$TB \
  data.max_prompt_length=1024 \
  data.max_response_length=$MAXRESP \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MB \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents='[model,hf_model]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=$G \
  actor_rollout_ref.rollout.temperature=$TEMP \
  actor_rollout_ref.rollout.top_p=$TOPP \
  actor_rollout_ref.rollout.top_k=$TOPK \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  reward.custom_reward_function.path="$REPO/src/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.logger='[console]' \
  trainer.total_training_steps=20 \
  trainer.total_epochs=100 \
  trainer.save_freq=5 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.project_name=frontier-teacher \
  trainer.experiment_name="$EXP" \
  trainer.default_local_dir="$CKPT" \
  ${THINK:+ +data.apply_chat_template_kwargs.enable_thinking=$THINK} \
  "${@:4}"
