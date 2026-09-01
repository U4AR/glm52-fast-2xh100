#!/usr/bin/env bash
# ============================================================================
# config.sh — THE ONE PLACE to configure paths for this project.
#
# Every launcher (run_fast.sh, run_server.sh, run_server_int4.sh) and the
# bench/experiment wrappers source this file, so changing a path here changes
# it everywhere. Every value is ALSO overridable per-invocation via env:
#   WEIGHTS_DIR=/mnt/nvme/glm ./run_fast.sh      # move all weights at once
#   MODEL=/some/other/GLM-5.2-W4AFP8 ./run_fast.sh
#
# Defaults are REPO-RELATIVE so a fresh clone runs with no edits: put the
# checkpoints under ./weights/ (a real directory, or a symlink to fast NVMe/
# scratch) and go. The download script (int4_scripts/download_w4afp8.py) writes
# there by default.
# ============================================================================

# Absolute path to this repo, resolved from THIS file's own location so the
# project can be cloned anywhere.
_CONFIG_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$_CONFIG_REPO}"

# ---- edit paths here (or override any of them via env) ----------------------
# Parent directory that holds the downloaded checkpoints. May be a symlink to
# fast scratch, e.g.  ln -s /cache/nvme0/models "$REPO/weights"
export WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO/weights}"

# The INT4 (W4AFP8) checkpoint — the fast default recipe (run_fast / int4).
export W4AFP8_MODEL="${W4AFP8_MODEL:-$WEIGHTS_DIR/GLM-5.2-W4AFP8}"
# The FP8 checkpoint — the baseline recipe (run_server.sh).
export FP8_MODEL="${FP8_MODEL:-$WEIGHTS_DIR/GLM-5.2-FP8}"
# Optional GPTQ-repacked INT4 CPU experts (int4_scripts/gptq_full_repack.py).
export GPTQ_EXPERTS_DIR="${GPTQ_EXPERTS_DIR:-$WEIGHTS_DIR/GLM-5.2-W4-GPTQ-experts}"

# ---- runtime locations (repo-local so nothing leaks onto the root disk) -----
export VENV="${VENV:-$REPO/.venv}"
export HF_HOME="${HF_HOME:-$REPO/.hf}"

# Gated Subspace Inference is opt-in.  The runtime reads these directly; keeping
# them centralized here makes every launcher and benchmark use the same mode.
export GSI_MODE="${GSI_MODE:-off}"
export GSI_IMAGE_DTYPE="${GSI_IMAGE_DTYPE:-bf16}"
export GSI_STRICT_FALLBACK="${GSI_STRICT_FALLBACK:-1}"
export GSI_CACHE_DIR="${GSI_CACHE_DIR:-$REPO/gsi_artifacts}"
