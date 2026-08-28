#!/usr/bin/env bash
# Idempotent environment bootstrap. Safe to re-run on a node already set up.
#
#   ./scripts/setup_env.sh              install from requirements.txt
#   LOCK=1 ./scripts/setup_env.sh       install the exact pinned set instead
set -eu
cd "$(dirname "$0")/.."

ENV_NAME=frontier-teacher
REQ=requirements.txt
[ "${LOCK:-0}" = "1" ] && REQ=requirements.lock.txt

if [ ! -d "$HOME/miniforge3" ]; then
  echo "==> installing miniforge"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm -f /tmp/miniforge.sh
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"

if python -c "import vllm" 2>/dev/null; then
  echo "==> vllm already present, skipping install (delete the env to force a rebuild)"
else
  echo "==> installing from $REQ"
  pip install --upgrade pip
  # vllm first and alone: it selects the CUDA-matched torch build, and letting a
  # later resolution step move torch is what breaks this environment.
  grep -E '^vllm==' "$REQ" | xargs pip install
  pip install -r "$REQ"
fi

python - <<'PY'
import importlib.metadata as m
for p in ("vllm", "torch", "transformers", "math-verify", "datasets"):
    print(f"  {p}=={m.version(p)}")
PY
echo "SETUP_DONE_OK"
