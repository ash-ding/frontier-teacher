#!/usr/bin/env bash
# One GRPO run: train/grpo/run_grpo.sh <config-name> <band-slug> [n_gpus]
#
#   train/grpo/run_grpo.sh llama32-3b pass1_05-15pct
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
cd "$(dirname "$0")/../.."
REPO="$PWD"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate frontier-teacher

CFG="${1:?usage: run_grpo.sh <config-name> <band-slug> [n_gpus]}"
BAND="${2:?usage: run_grpo.sh <config-name> <band-slug> [n_gpus]}"
NGPU="${3:-8}"

# max_response_length must not bind tighter than evaluation does, or training and
# evaluation are not measuring the same model. Evaluation allows 30,000 tokens
# (configs/*.yaml max_tokens), and a first thinking run capped at 12,288 had 52%
# of its rollouts truncated - rising from 31% at step 1 to 60% by step 20 as GRPO
# lengthened responses. A truncated rollout carries no \boxed{} and scores 0, so
# over half the batch was being graded on where the cap fell rather than on
# whether the answer was right. 32,768 restores the match with evaluation.
case "$CFG" in
  llama32-3b)       MODEL=unsloth/Llama-3.2-3B-Instruct; MAXRESP=4096;  TEMP=0.6; TOPP=0.9;  TOPK=-1; THINK='';    MEMUTIL=0.5 ;;
  qwen3-4b-nothink) MODEL=Qwen/Qwen3-4B;                 MAXRESP=8192;  TEMP=0.7; TOPP=0.8;  TOPK=20; THINK=false; MEMUTIL=0.5 ;;
  qwen3-4b-think)   MODEL=Qwen/Qwen3-4B;                 MAXRESP=32768; TEMP=0.6; TOPP=0.95; TOPK=20; THINK=true;  MEMUTIL=0.45 ;;
  *) echo "unknown config $CFG"; exit 1 ;;
esac

# A fixed micro-batch COUNT cannot express a 32k cap: two 33,792-token sequences
# in one backward pass is ~2.5x the activation memory that fit at 12k, and the
# same count is wasteful when a sequence comes back at 1,700 tokens. Batch by
# token budget instead - identical worst case (one maximal sequence) with the
# short ones packed densely. The budget must exceed prompt+response or verl
# cannot place the longest sequence at all.
# The budget was 34,816 (prompt + response + slack) at vLLM 0.6, and the low
# thinking band died of CUDA OOM at step 19 of 20: the logits tensor alone is
# tokens x 151,936 vocab x 2 bytes = 10.6 GB, the temperature division makes a
# second copy, and vLLM still held 35.9 GB. 11.63 GB was wanted with 11.03 free.
# Trim the budget to just above the longest possible sequence and give vLLM less,
# which costs some rollout concurrency and buys the run finishing.
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

# middle band -> G=8; the two extreme bands -> G=32. Both give 512 rollouts/step.
case "$BAND" in
  *37-62pct*|*40-60pct*) G=8;  TB=64; MB=32 ;;
  *)                     G=32; TB=16; MB=8  ;;
esac

# GROUP_SIZE forces G, overriding the band default, with the train batch derived
# to keep 512 rollouts per step. This exists because tuning G per band leaves the
# band comparison confounded: at a fixed rollout budget G also fixes how many
# distinct problems a step covers (512/G), so the middle band at G=8 saw 64
# problems per step against the extremes' 16 — four times the data over a run,
# and correspondingly more epochs. Re-running a middle band at GROUP_SIZE=32
# matches it to the extremes on every axis except the difficulty band itself.
#
# It costs nothing in zero-gradient groups. The (1-p)^G + p^G waste that drives
# the default mapping is 48.2% for G=8 at p=8.7% but 0% for G=32 at p=49.5% —
# G=32 on a middle band is statistically over-sampled, not wasteful. What it
# spends is problem diversity: the same 10,240 rollouts over 320 problem
# instances instead of 1,280.
if [ -n "${GROUP_SIZE:-}" ]; then
  G=$GROUP_SIZE; TB=$((512 / G)); MB=$((TB / 2))
  echo "  GROUP_SIZE override: G=$G train_batch=$TB mini_batch=$MB"
fi

SUB=$(ls data/further_improve/"$CFG"/"$CFG"__"$BAND"__n*.jsonl 2>/dev/null | head -1)
[ -n "$SUB" ] || { echo "no subset matching $CFG/$BAND"; exit 1; }
TRAIN="${SUB%.jsonl}.verl.jsonl"
[ -f "$TRAIN" ] || python train/grpo/to_verl_dataset.py --subset "$SUB"

EXP="${CFG}__${BAND}${TAG:-}"
# Checkpoints go to the LOCAL container disk, not ~/data. One checkpoint is 26 GB
# with optimizer state dropped, so the 36 across all runs would be ~940 GB of
# writes over the rclone bucket mount. The orchestrator evaluates each run's
# checkpoints on the same node and deletes them before the next run starts;
# per-run peak is 4 x 26 GB = 104 GB against 174 GB free.
CKPT="$REPO/outputs/checkpoints/$EXP"

# Start from the base model, always. verl's default resume_mode=auto silently
# picks up whatever checkpoint directory it finds and continues from it - which
# on a re-run means training resumes from an ABORTED earlier attempt whose step
# count and provenance are unknown, and whose curve is then not comparable with
# the other eight cells. It also interacts badly with reclaiming disk: the
# janitor that strips redundant FSDP shards left a checkpoint that auto-resume
# could see but not load, and the run died at startup with FileNotFoundError.
# Each of the nine runs is a clean 20 steps from the released weights.
if [ -d "$CKPT" ] && [ -n "$(ls -d "$CKPT"/global_step_* 2>/dev/null)" ]; then
  echo "  clearing stale checkpoints in $CKPT"
  rm -rf "$CKPT"
fi
mkdir -p "$CKPT" logs

echo "=== $EXP ==="
echo "  model=$MODEL  subset=$(basename "$SUB")  problems=$(wc -l < "$SUB")"
echo "  G=$G  train_batch=$TB  mini_batch=$MB  -> $((TB*G)) rollouts/step, 20 steps = $((TB*G*20))"
echo "  max_response=$MAXRESP  enable_thinking=${THINK:-n/a}  gpus=$NGPU  vllm_util=$MEMUTIL"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$REPO/$TRAIN" \
  data.val_files="$REPO/$TRAIN" \
  data.train_batch_size=$TB \
  data.max_prompt_length=1024 \
  data.max_response_length=$MAXRESP \
  actor_rollout_ref.model.path="$MODEL" \
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
  reward.custom_reward_function.path="$REPO/train/grpo/verl_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.use_v1=False \
  trainer.resume_mode=disable \
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
