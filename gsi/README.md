# GLM-5.2 Gated Subspace Inference

Implementation of **arXiv 2605.03109**, *Gated Subspace Inference*, on the
GLM-5.2 W4AFP8 serving stack.

The method fits an orthonormal basis `V` to the *activations* entering each
linear map, caches the image `M = W V`, and gates per token on the residual
ratio `rho = ||x - V V' x|| / ||x||`.  Rows with `rho < eps` take the fast path
`y = M (V' x)`; the rest run the exact quantized map.  **The fast path drops the
`W r` correction**, so it is approximate by construction -- the paper reports
98.6% top-1 agreement at its headline operating point, not exact tokens.  Score
runs with `gsi.evaluate`, never with exact greedy equality.

## Modes

- `off`: no imports or behavior changes in model execution.
- `observe`: exact baseline output plus bounded activation reservoirs. CUDA
  graphs are disabled automatically.
- `functional`: cached-image fast paths with exact baseline fallback for
  standard attention/dense/shared linear modules. CUDA graphs are disabled.
- `kernel`: reserved for the fused, CUDA-graph-safe routed-expert backend and
  rejected unless `GSI_EXPERIMENTAL_KERNEL=1` is set.

## Coverage

Every linear map the paper covers has a calibrated basis:

| site | module | notes |
|---|---|---|
| QKV | `fused_qkv_a_proj_with_mqa` | reads the residual stream |
| q latent | `q_b_proj` | |
| attention out | `o_proj` | reads the head concatenation |
| dense/shared MLP | `gate_up_proj`, `down_proj` | layers 0-2 dense, 3+ shared expert |
| routed experts | `mlp.routed_gate_up`, `mlp.routed_down` | 256 experts x 75 layers |

`kv_b_proj` is absent because MLA absorbs it into `w_kc`/`w_vc` and it never runs
as a linear.  Routed experts are consumed by a fused grouped GEMM rather than a
`LinearBase`, so their images are built by `gsi.expert_images` through the same
CUTLASS kernel the slow path uses, and their post-SiLU input is captured from
inside `cutlass_w4a8_moe` using the layer id published by `runtime.routed_layer`.

## Basis scope and depth cascade

The paper fits **one basis per layer**, shared across every map in that layer.
`--basis-scope layer` (the default) does this, keyed by activation width: in
GLM-5.2 the residual stream (6144), the head concatenation (8192), the q-latent
(2048) and the SwiGLU intermediates are different spaces and cannot share a
basis, so the residual-stream maps -- `fused_qkv_a`, `gate_up_proj` and
`routed_gate_up` -- share one basis per layer and the rest get their own.
`--basis-scope module` restores an independent basis per map.

`--cascade` seeds each layer's basis from the previous layer and refines it with
orthogonal iteration instead of taking an independent thin SVD (the paper's 96%
calibration-cost reduction).  Depth coherence is measured either way and written
to `depth-coherence.json` as the mean cosine of principal angles between
consecutive layers -- the paper observes >0.90 past layer 8 on GPT-J.

## Calibration flow

```bash
# 1. Capture exact decode/verify activations.
GSI_MODE=observe GSI_CAPTURE_DIR=$PWD/gsi_captures ./run_fast.sh

# 2. Build bases and residual sweeps.  Defaults: eps=0.10, per-layer basis.
python -m gsi.calibrate \
  --captures gsi_captures/*.safetensors \
  --out gsi_artifacts/calibration \
  --model-hash <checkpoint-manifest-hash> \
  --ranks 256 512 --cascade

# 3. Start once with the quantized model loaded to build WV through the actual
#    deployed quantization methods. Each TP rank writes its own image shard.
GSI_MODE=functional \
GSI_PROFILE=$PWD/gsi_artifacts/calibration/profile.json \
GSI_CACHE_DIR=$PWD/gsi_artifacts/calibration \
GSI_BUILD_IMAGES=1 ./run_fast.sh

# 4. Use the resulting profile.
GSI_MODE=functional \
GSI_PROFILE=$PWD/gsi_artifacts/calibration/profile.built.json ./run_fast.sh
```

Finalize and checksum every TP shard after image construction:

```bash
python -m gsi.finalize \
  --profile gsi_artifacts/calibration/profile.built.json \
  --out gsi_artifacts/calibration/profile.final.json
```

## Scoring

```bash
# Freeze baseline and candidate traces with per-token logprobs.
python bench/gsi/capture_endpoint.py --label baseline --logprobs \
  --out gsi_reports/baseline.json
python bench/gsi/capture_endpoint.py --label gsi --logprobs \
  --out gsi_reports/candidate.json

python -m gsi.evaluate \
  --profile gsi_artifacts/calibration/profile.final.json \
  --telemetry gsi_reports/run.tp0.json \
  --baseline-trace gsi_reports/baseline.json \
  --candidate-trace gsi_reports/candidate.json \
  --out gsi_reports/score.json
```

`gsi.evaluate` reports the paper's columns -- perplexity ratio, top-1 agreement,
generation agreement, fast-path fraction and `S_eff` from Eq. 6 -- alongside
`deployed_speedup`, which is the same run scored on batch-1 weight *bytes*
against the W4AFP8 path actually served.  The two differ by a large factor and
must not be conflated: Eq. 6 assumes a dense BF16 baseline and a free
projection, giving a `d/k` ceiling; against packed INT4 weights the ceiling is
`d/(4k)` and the `d x k` projection is paid on every token.  The paper evaluates
no quantized baseline (S7.4 lists it as future work).

The negative control the paper requires -- static projection, residual discarded
unconditionally -- is `GSI_GATE_OVERRIDE=always_fast`; `all_slow` forces the
exact path for the fidelity reference.

Estimate cache and host-weight storage directly from checkpoint metadata:

```bash
python -m gsi.footprint \
  --model /path/to/GLM-5.2-W4AFP8 \
  --gate-rank 256 --down-rank 256 --image-dtype fp8
```

Measurements and the stop/go decision are in `GSI_STATUS.md`.
