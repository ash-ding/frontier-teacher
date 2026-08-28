#!/usr/bin/env bash
# Build the RL training environment, SEPARATE from the evaluation environment.
#
# They cannot share one env. verl depends on hydra-core, which pins
# antlr4-python3-runtime==4.9.*, while math-verify[antlr4_13_2] pins ==4.13.2.
# pip resolves that by downgrading, and the wrong antlr4 makes math-verify's
# LaTeX parser fail silently - it does not raise, it just stops parsing and
# depresses every score. verl also pulls transformers back from 5.16 to 5.10.
#
# So: frontier-teacher evaluates, frontier-teacher-rl trains. Same torch and
# vllm in both (2.13.0 / 0.27.1), so a checkpoint moves between them cleanly.
set -eu
cd "$(dirname "$0")/.."

ENV_NAME=frontier-teacher-rl
REQ=requirements-rl.txt

[ -d "$HOME/miniforge3" ] || { echo "run scripts/setup_env.sh first (installs miniforge)"; exit 1; }
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"

if python -c "import verl" 2>/dev/null; then
  echo "==> verl already present, skipping install"
else
  pip install --upgrade pip
  # vllm first and alone so it fixes the CUDA-matched torch build; verl is
  # installed against that rather than being allowed to move it.
  pip install "vllm==0.27.1"
  pip install "verl==0.9.0"
  pip install "flash-attn==2.8.3.post1" --no-build-isolation || \
    echo "WARNING: flash-attn build failed; verl runs without it, slower"
fi

python - <<'PY'
import importlib.metadata as m
for p in ("verl", "torch", "vllm", "transformers", "ray", "tensordict", "peft"):
    try: print(f"  {p}=={m.version(p)}")
    except Exception: print(f"  {p}  MISSING")
PY
echo "RL_SETUP_DONE_OK"
