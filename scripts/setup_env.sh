#!/usr/bin/env bash
# Idempotent environment bootstrap. Safe to re-run on a host that is already set up.
set -eux

if [ ! -d "$HOME/miniforge3" ]; then
  cd "$HOME"
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  rm -f /tmp/miniforge.sh
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^frontier-teacher " || conda create -y -n frontier-teacher python=3.12
conda activate frontier-teacher

python -c "import vllm" 2>/dev/null || {
  pip install --upgrade pip
  pip install "vllm==0.27.1"
  pip install "math-verify[antlr4_13_2]" datasets transformers accelerate pandas tabulate pyyaml
}
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
echo "SETUP_DONE_OK"
