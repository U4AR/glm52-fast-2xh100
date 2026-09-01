#!/usr/bin/env bash
# Bootstrap an isolated GatedGLM runtime without reading or modifying RunGLM.
set -euo pipefail
cd "$(dirname "$0")"

source ./config.sh

KT_REPO=${KT_REPO:-https://github.com/U4AR/ktransformers.git}
KT_BRANCH=${KT_BRANCH:-glm5.2-2xh100-stable}
KT_COMMIT=${KT_COMMIT:-8d83454}

if [ ! -d ktransformers/.git ]; then
  git clone --recursive --branch "$KT_BRANCH" "$KT_REPO" ktransformers
fi
git -C ktransformers fetch origin "$KT_BRANCH"
git -C ktransformers checkout "$KT_COMMIT"
git -C ktransformers submodule update --init --recursive

if [ ! -x "$VENV/bin/python" ]; then
  python3.12 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip

export PKG_CONFIG_PATH="$VENV/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$VENV:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$VENV/lib:${LD_LIBRARY_PATH:-}"

CPUINFER_USE_CUDA=1 ./ktransformers/install.sh
python -m pip install -r requirements-lock.txt

# Restore the tracked GLM/MTP/GSI overlays that the clean install replaces.
git checkout -- .venv

python - <<'PY'
import torch
import sglang
import kt_kernel
print("torch", torch.__version__)
print("sglang", sglang.__file__)
print("kt_kernel", kt_kernel.__file__)
PY

echo "GatedGLM runtime ready at $VENV"
