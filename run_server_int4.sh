#!/usr/bin/env bash
# Launch GLM-5.2-W4AFP8 (INT4 experts + FP8 non-experts) on 2x H100 NVL (TP2)
# + KT-Kernel CPU-GPU heterogeneous MoE, with INT4 (RAWINT4) CPU+GPU experts.
#
# Route: kt RAWINT4. The W4AFP8 on-disk int4 payload is byte-identical to kt's
# native compressed-tensors layout (verified: lo-nibble-first signed int4,
# 2-packed along K, bf16 group-128 scale, symmetric). Only tensor names differ
# (.weight/.weight_scale_inv vs .weight_packed/.weight_scale), handled by a
# patch to kt CompressedSafeTensorLoader (no disk repack). The fp8 activation
# input_scale is unused by RAWINT4. Non-expert weights are block-FP8 -> sglang
# Fp8LinearMethod (W4AFp8Config routes LinearBase->fp8, experts->kt).
#
# This box: 2x H100 NVL (96GB), AMD EPYC 9V84 (80c, 2 NUMA, AVX512 no-AMX), 629GB.
set -euo pipefail

# All paths come from config.sh (the one place to edit them); every value is
# still overridable via env. See config.sh for WEIGHTS_DIR / W4AFP8_MODEL / etc.
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
# sglang reads non-expert (block-FP8/BF16) weights from MODEL; kt reads the INT4
# routed experts from KT_WEIGHT_PATH (separate dir for the GPTQ-repacked experts).
# WINNING RECIPE (2026-06-25, ~10.1-10.6 tok/s decode > 8.7 FP8 baseline):
#   int4 GPU experts (MODEL=W4AFP8 -> W4AFp8MoEMethod cutlass) + FP8 CPU experts
#   (fast AVX512) + GPU_EXPERTS=104 (88GB/card). Needs the w4afp8.py -1-remap fix.
#   GPTQ_INT4 CPU (nvme1) boots faster (~3min vs ~50min) but is AVX2-slow (~6.4).
# Point MODEL/KT_WEIGHT_PATH at your downloaded weights (default: ./weights/...).
MODEL=${MODEL:-$W4AFP8_MODEL}
# This checkpoint's stable host fallback is the native packed INT4 payload.
# The FP8 host loader is not equivalent on this machine and can segfault while
# materializing cold experts (reproduced at layers 9/77 for 96/104 hot experts).
KT_METHOD=${KT_METHOD:-RAWINT4}
KT_WEIGHT_PATH=${KT_WEIGHT_PATH:-$MODEL}
PORT=${PORT:-8000}
RANDOM_SEED=${RANDOM_SEED:-}
DETERMINISTIC=${DETERMINISTIC:-0}

# OpenAI-compatible /v1/chat/completions needs a chat template. The W4AFP8 dir
# ships no tokenizer.chat_template, so point sglang at the repo-local GLM jinja
# (renders the reasoning-effort system prompt + tool-call format the glm45/glm47
# parsers expect). Override CHAT_TEMPLATE= to disable.
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$REPO/chat_template.jinja}

source "$VENV/bin/activate"

# Make the repo-native GSI package visible to scheduler/model worker processes.
# The installed SGLang patch imports it lazily only when GSI_MODE != off.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ "$GSI_MODE" = "observe" ] && [ -z "${GSI_CAPTURE_DIR:-}" ]; then
  export GSI_CAPTURE_DIR="$REPO/gsi_captures"
fi
if { [ "$GSI_MODE" = "functional" ] || [ "$GSI_MODE" = "kernel" ]; } && [ -z "${GSI_PROFILE:-}" ]; then
  echo "ERROR: GSI_PROFILE is required for GSI_MODE=$GSI_MODE" >&2
  exit 2
fi

# --- runtime env -----------------------------------------------------------
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$REPO/.hf}
mkdir -p "$HF_HOME"
# RAWINT4 backend on this no-AMX EPYC: default selection picks AMXInt4_KGroup_MOE
# (avx512_bf16-compiled). If it faults with an illegal instruction, force AVX2:
#   export KT_RAWINT4_BACKEND=avx2
if [ "$KT_METHOD" = "RAWINT4" ]; then
  export KT_RAWINT4_BACKEND=${KT_RAWINT4_BACKEND:-avx512_packed}
elif [ -n "${KT_RAWINT4_BACKEND:-}" ]; then
  export KT_RAWINT4_BACKEND
fi

# --- tunables --------------------------------------------------------------
# INT4 experts are ~half the bytes of FP8 -> less VRAM/card AND less CPU RAM,
# so GPU_EXPERTS can go HIGHER than the FP8 run's 48. Start conservative; tune up.
# 96 is the safe default (82GB/card, fits 8192-token KV, ~9.8 tok/s). For the
# max squeeze use GPU_EXPERTS=104 MAX_TOTAL_TOKENS=4096 (88GB/card, ~10.5). 112 OOMs.
GPU_EXPERTS=${GPU_EXPERTS:-96}
MEM_FRACTION=${MEM_FRACTION:-0.94}
CPUINFER=${CPUINFER:-72}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-8192}
# Explicit max sequence length (positions). Model supports 1M, but leave unset
# and sglang derives a huge default; pin it so a coding agent's long context is
# admitted predictably and the KV pool is sized to match MAX_TOTAL_TOKENS.
CONTEXT_LENGTH=${CONTEXT_LENGTH:-}
MAX_RUNNING=${MAX_RUNNING:-2}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-2048}
# GPU bulk prefill: a prefill chunk with >= this many tokens streams ALL 256
# experts to GPU (cutlass W4A8) for the chunk instead of computing the CPU
# experts on the AVX-512 path. Measured ~1.9x faster TTFT on big prompts (2660
# tok: ~20s vs ~37s CPU), decode unaffected (~14 tok/s). Default 2048 = the chunk
# size, so only full chunks take the GPU path (>2x the ~860-token break-even).
# Requires the rebuilt kt-kernel (packed RAWINT4 write_weights_to_buffer, which
# the upstream backend left as a "not yet implemented" stub) and ~5 GB of free
# VRAM for the transient 256-expert scratch layer (fits at GPU_EXPERTS=96/128k,
# avail≈8 GB). Set KT_GPU_PREFILL_THRESHOLD=0 to disable (pure CPU prefill).
KT_GPU_PREFILL_THRESHOLD=${KT_GPU_PREFILL_THRESHOLD:-2048}
DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-1}
SLEEP_ON_IDLE=${SLEEP_ON_IDLE:-1}

# Observe/functional mode contains Python-side capture/gating and is deliberately
# eager.  The production graph path is capability-gated by GSI_EXPERIMENTAL_KERNEL.
if [ "$GSI_MODE" != "off" ] && [ "$GSI_MODE" != "kernel" ] && [ "${GSI_ALLOW_CUDA_GRAPH:-0}" != "1" ]; then
  DISABLE_CUDA_GRAPH=1
fi

CG_FLAG=""
if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
  CG_FLAG="--disable-cuda-graph"
  export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
else
  export SGLANG_ENABLE_JIT_DEEPGEMM=1
  CG_FLAG="--cuda-graph-max-bs $CUDA_GRAPH_MAX_BS --disable-custom-all-reduce"
fi

DYN_UPDATE=${DYN_UPDATE:-0}
DYN_FLAG=""; [ "$DYN_UPDATE" = "1" ] && DYN_FLAG="--kt-enable-dynamic-expert-update"

IDLE_FLAG=""; [ "$SLEEP_ON_IDLE" = "1" ] && IDLE_FLAG="--sleep-on-idle"

# Debug: NaN detection (localizes garbage to a layer/op) + force all experts to
# CPU (GPU_EXPERTS=0) to bisect CPU-int4 vs GPU-int4 kernel.
NAN_DETECT=${NAN_DETECT:-0}
NAN_FLAG=""; [ "$NAN_DETECT" = "1" ] && NAN_FLAG="--enable-nan-detection"

# --- MTP / NEXTN (OFF for first int4 boot; turn on after baseline) ---------
SPEC_DECODE=${SPEC_DECODE:-0}
SPEC_STEPS=${SPEC_STEPS:-1}
SPEC_TOPK=${SPEC_TOPK:-1}
SPEC_DRAFT_TOKENS=${SPEC_DRAFT_TOKENS:-2}
# decode = draft-extend uses the decode attention backend (flashmla needs
# FlashMLADecodeMetadata.block_kv_indices; the default 'prefill' mode hands the
# draft a PrefillMetadata and crashes in flashmla_backend.forward_extend).
SPEC_ATTN_MODE=${SPEC_ATTN_MODE:-decode}
# flashmla's draft-extend path is broken in this build (hands the draft a
# PrefillMetadata but runs the decode kernel that needs block_kv_indices), so
# run the DRAFT's attention on fa3 (the standard EAGLE Hopper backend) while the
# TARGET model stays on flashmla. Set SPEC_DRAFT_ATTN= to disable the override.
SPEC_DRAFT_ATTN=${SPEC_DRAFT_ATTN-fa3}
SPEC_FLAG=""
if [ "$SPEC_DECODE" = "1" ]; then
  export SGLANG_ENABLE_SPEC_V2=${SGLANG_ENABLE_SPEC_V2:-True}
  SPEC_FLAG="--speculative-algorithm NEXTN --speculative-num-steps $SPEC_STEPS --speculative-eagle-topk $SPEC_TOPK --speculative-num-draft-tokens $SPEC_DRAFT_TOKENS --speculative-attention-mode $SPEC_ATTN_MODE"
  [ -n "$SPEC_DRAFT_ATTN" ] && SPEC_FLAG="$SPEC_FLAG --speculative-draft-attention-backend $SPEC_DRAFT_ATTN"
fi

# --- NSA attention sub-backends (override for spec-verify experiments) ------
# target_verify is extend-like -> uses the PREFILL backend. Default (fp8 KV) is
# flashmla_auto; swap to flashmla_kv/fa3/etc to test the spec-verify garbage.
NSA_PREFILL_BACKEND=${NSA_PREFILL_BACKEND:-}
NSA_DECODE_BACKEND=${NSA_DECODE_BACKEND:-}
NSA_FLAG=""
[ -n "$NSA_PREFILL_BACKEND" ] && NSA_FLAG="$NSA_FLAG --nsa-prefill-backend $NSA_PREFILL_BACKEND"
[ -n "$NSA_DECODE_BACKEND" ] && NSA_FLAG="$NSA_FLAG --nsa-decode-backend $NSA_DECODE_BACKEND"

CHAT_TEMPLATE_FLAG=""
[ -n "$CHAT_TEMPLATE" ] && CHAT_TEMPLATE_FLAG="--chat-template $CHAT_TEMPLATE"

CONTEXT_LENGTH_FLAG=""
[ -n "$CONTEXT_LENGTH" ] && CONTEXT_LENGTH_FLAG="--context-length $CONTEXT_LENGTH"

RANDOM_SEED_FLAG=""
[ -n "$RANDOM_SEED" ] && RANDOM_SEED_FLAG="--random-seed $RANDOM_SEED"
DETERMINISTIC_FLAG=""
[ "$DETERMINISTIC" = "1" ] && DETERMINISTIC_FLAG="--enable-deterministic-inference"

# --- NSA (Native Sparse Attention) long-context fix --------------------------
# GLM-5.2 uses DeepSeek Sparse Attention: a lightning indexer selects the top
# `index_topk` (=2048) tokens per query once the sequence exceeds 2048. This
# sglang build runs EVERY layer through that sparse path (no per-layer Full/Sparse
# `index_topk_pattern` support, which newer sglang has), and the sparse path
# produces GARBAGE beyond 2048 tokens (hard cliff: 2041 ok, 2061 gibberish). The
# topk kernels also hard-assert topk==2048, so widening the dense window is not
# possible. The documented + optimal fix for our box: disable NSA entirely and run
# full DENSE MLA attention. is_deepseek_nsa() is gated on `index_topk is not None`
# (model_config.py), so overriding index_topk=null turns NSA off cleanly -> plain
# MLA. Dense attention is the most ACCURATE (sparse only approximates it) and
# costs us ~nothing: we are CPU-MoE bound (~13 tok/s, GPUs ~50% idle), so the
# extra attention FLOPs are hidden. DISABLE_NSA=1 (default) also forces a dense
# MLA attention backend. Set DISABLE_NSA=0 to restore native NSA (buggy >2048).
DISABLE_NSA=${DISABLE_NSA:-1}
MODEL_OVERRIDE_FLAG=""
if [ "$DISABLE_NSA" = "1" ]; then
  MODEL_OVERRIDE_FLAG='--json-model-override-args {"index_topk":null}'
  ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashmla}
else
  ATTENTION_BACKEND=${ATTENTION_BACKEND:-nsa}
fi
# Optional page-size override (the 'compressed'/DeepseekV4 in-graph-metadata
# backend needs page_size 256; flashmla forces 64).
PAGE_SIZE_FLAG=""
[ -n "${PAGE_SIZE:-}" ] && PAGE_SIZE_FLAG="--page-size $PAGE_SIZE"

echo "GLM-5.2-$KT_METHOD  TP${TP_SIZE:-2}  model=$MODEL  kt_weights=$KT_WEIGHT_PATH  port=$PORT  gpu_experts=$GPU_EXPERTS  mem_fraction=$MEM_FRACTION  cpuinfer=$CPUINFER  cuda_graph=$([ "$DISABLE_CUDA_GRAPH" = 1 ] && echo off || echo on)  sleep_on_idle=$SLEEP_ON_IDLE  spec_decode=$([ "$SPEC_DECODE" = 1 ] && echo on || echo off)  rawint4_backend=${KT_RAWINT4_BACKEND:-auto}  nsa_prefill=${NSA_PREFILL_BACKEND:-default}  gsi=$GSI_MODE"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --kt-weight-path "$KT_WEIGHT_PATH" \
  --kt-cpuinfer "$CPUINFER" \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts "$GPU_EXPERTS" \
  --kt-method "$KT_METHOD" \
  --kt-gpu-prefill-token-threshold "${KT_GPU_PREFILL_THRESHOLD:-2048}" \
  $DYN_FLAG \
  --kt-expert-placement-strategy uniform \
  --tp-size "${TP_SIZE:-2}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  $RANDOM_SEED_FLAG \
  $DETERMINISTIC_FLAG \
  --mem-fraction-static "$MEM_FRACTION" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  --max-total-tokens "$MAX_TOTAL_TOKENS" \
  $CONTEXT_LENGTH_FLAG \
  $MODEL_OVERRIDE_FLAG \
  --max-running-requests "$MAX_RUNNING" \
  --chunked-prefill-size "$CHUNKED_PREFILL" \
  $IDLE_FLAG \
  $PAGE_SIZE_FLAG \
  $CG_FLAG \
  $SPEC_FLAG \
  $NAN_FLAG \
  --attention-backend "$ATTENTION_BACKEND" \
  $NSA_FLAG \
  $CHAT_TEMPLATE_FLAG \
  --fp8-gemm-backend cutlass \
  --disable-shared-experts-fusion \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name GLM5.2
