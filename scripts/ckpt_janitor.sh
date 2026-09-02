#!/usr/bin/env bash
# verl writes each checkpoint twice: FSDP shards for resuming (16 GB) and an HF
# export for loading (15 GB). resume_mode is disabled on every run, so only the
# export is ever read and the shards are pure disk cost. Strip them once settled.
#
# Installed via cron (*/5) rather than a background loop: a loop doing this died
# silently on one node and let 122 GB accumulate against 53 GB free.
find "$HOME/code/frontier-teacher/.local_checkpoints" \
     -name 'model_world_size_*.pt' -mmin +3 -delete 2>/dev/null
exit 0
