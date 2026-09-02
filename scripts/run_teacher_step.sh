#!/usr/bin/env bash
# One GRPO update for the observational teacher loop.
#
#   scripts/run_teacher_step.sh <config-name> <train_file> <model_path> <ckpt_dir> [n_gpus]
#
# Sibling of run_grpo.sh, kept separate so that validated script stays untouched.
# It reuses run_grpo.sh's llama32-3b preset values VERBATIM (MODEL default,
# MAXRESP/TEMP/TOPP/TOPK/MEMUTIL, the <=16k micro-batch branch, and the extreme-
# band G=32/TB=16/MB=8 group sizing that gives 512 rollouts) and the same
# main_ppo invocation. Only the loop-specific knobs differ:
#   * data.train_files / data.val_files -> this step's teacher-authored verl file
#   * actor_rollout_ref.model.path      -> base model (step 0) or the prior hf ckpt
#   * trainer.resume_mode=disable       -> weights come from model.path, not a saved
#                                          dataloader/optimizer state (verl silently
#                                          desyncs its cursor if train_files change
#                                          under resume_path)
#   * trainer.total_training_steps=1    -> exactly one optimizer step over the batch
#   * trainer.save_freq=1               -> save the checkpoint at step 1
#   * trainer.default_local_dir         -> this step's ckpt dir
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: run_teacher_step.sh <config-name> <train_file> <model_path> <ckpt_dir> [n_gpus]}"
TRAIN="${2:?missing <train_file>}"
MODEL_PATH="${3:?missing <model_path>}"
CKPT="${4:?missing <ckpt_dir>}"
NGPU="${5:-8}"

# --- llama32-3b preset, copied verbatim from run_grpo.sh -----------------------
case "$CFG" in
  llama32-3b)       MODEL=unsloth/Llama-3.2-3B-Instruct; MAXRESP=4096;  TEMP=0.6; TOPP=0.9;  TOPK=-1; THINK='';    MEMUTIL=0.5 ;;
  qwen3-4b-nothink) MODEL=Qwen/Qwen3-4B;                 MAXRESP=8192;  TEMP=0.7; TOPP=0.8;  TOPK=20; THINK=false; MEMUTIL=0.5 ;;
  qwen3-4b-think)   MODEL=Qwen/Qwen3-4B;                 MAXRESP=32768; TEMP=0.6; TOPP=0.95; TOPK=20; THINK=true;  MEMUTIL=0.45 ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

SEQ=$((MAXRESP + 1024 + 256))
if [ "$MAXRESP" -gt 16384 ]; then
  BATCHING=(
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$SEQ
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((SEQ * 2))
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((SEQ * 2))
  )
else
  BATCHING=(
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  )
fi

# Teacher train steps use the extreme-band sizing: G=32 rollouts/problem. TB is the
# whole authored set (the teacher may write fewer than 16) so exactly one optimizer
# step covers every problem -> G*TB rollouts, one update. verl requires
# train_batch_size <= dataset size, so TB tracks N rather than a fixed 16.
G=32
# --- end verbatim preset -------------------------------------------------------

[ -f "$TRAIN" ] || { echo "missing train file $TRAIN"; exit 1; }
N=$(wc -l < "$TRAIN")
[ "$N" -ge 1 ] || { echo "empty train file $TRAIN"; exit 1; }
# mini_batch == train_batch: one full-batch gradient update over all N problems.
# verl requires train_batch_size % ppo_mini_batch_size == 0, and N is arbitrary
# (the teacher chooses how many problems to author), so equal sizing is the only
# choice that is always divisible -- and it is exactly the "one update on the
# batch" this loop wants.
TB=$N
MB=$N
mkdir -p "$CKPT" logs

echo "=== teacher step ==="
echo "  model_path=$MODEL_PATH  train=$TRAIN  problems=$N"
echo "  G=$G  train_batch=$TB  mini_batch=$MB  -> $((TB*G)) rollouts, 1 update"
echo "  max_response=$MAXRESP  gpus=$NGPU  vllm_util=$MEMUTIL  ckpt=$CKPT"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN" \
  data.val_files="$TRAIN" \
  data.train_batch_size=$TB \
  data.max_prompt_length=1024 \
  data.max_response_length=$MAXRESP \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MB \
  "${BATCHING[@]}" \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=$MEMUTIL \
  reward.custom_reward_function.path="$REPO/src/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node=$NGPU \
  trainer.nnodes=1 \
  trainer.logger='[console]' \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False \
  trainer.project_name=frontier-teacher-curriculum \
  trainer.experiment_name="teacher_step" \
  trainer.default_local_dir="$CKPT" \
  ${THINK:+ +data.apply_chat_template_kwargs.enable_thinking=$THINK} \
  "${@:6}"
