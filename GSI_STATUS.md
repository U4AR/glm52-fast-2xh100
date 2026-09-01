# GSI implementation and measurement status

Measured on 2026-07-29 in the isolated `GatedGLM` environment with GLM-5.2
W4AFP8, TP2, 104 GPU experts per routed layer, top-2 substitution, and NEXTN
depth-3.

## Implemented

- `off`, `observe`, `functional`, and capability-gated `kernel` modes.
- Decode/target-verification phase gating; exact prefill bypass.
- Stable residual-energy gating with invalid-input exact fallback.
- Per-module activation capture, GPU FP32 SVD calibration, versioned manifests,
  checksums, per-TP image shards, and telemetry.
- Quantized-operator `WV` construction for target attention, dense MLP, and
  shared-expert linears.
- Mixed fast/slow eager execution for standard quantized linears.
- Top-2/top-8 request compatibility with MTP enabled; the same launch controls
  preserve MTP-disabled operation.
- Semantic reference implementation for independently gated routed-expert
  gate/up and down stages.

The fused KT routed-expert image kernels, graph-captured conditional CPU
fallback, down-only KT task, and FP8 image format remain behind the kernel
capability gate.

## Generated artifact

The provisional rank-128 BF16 cache is under
`gsi_artifacts/rank128-top2-top8-mtp/` (ignored by git):

- 465 calibrated target entries: 390 standard linears and 75 routed gate/up
  inputs.
- 568 MiB of BF16 bases.
- 390 target images per TP rank, 780 images total, 977 MiB on disk.
- `profile.final.json` validates every basis and both TP image shards by SHA256.
- Routed-expert entries intentionally have no image file because the fused KT
  backend is not implemented.
- This capture predated the NEXTN phase-context fix, so the provisional profile
  does not contain NEXTN bases or images.

## Measurement gate

At rank 128 and epsilon 0.025:

- Calibration mean fast fraction across all captured inputs: **0.2877%**.
- Runtime target fast fraction:
  - TP0: 214 / 68,000 rows = **0.3147%**
  - TP1: 73 / 68,000 rows = **0.1074%**
- The largest TP0 fast site was layer 0 attention output: 139 / 176 rows.
- No invalid/NaN gate inputs were observed.
- Both tested greedy traces (top-2 + MTP and top-8 + MTP) diverged from their
  forced-all-slow controls.

The loaded target consumed about 83 GiB per H100, target KV used 3.43 GiB, and
NEXTN left only 0.68-0.70 GiB free per rank with the current 104-hot-expert and
81,920-token configuration. This is insufficient for the planned all-expert
image tier without first trading away hot full experts and/or context capacity.

## Rank 256/512 sweep (the follow-up experiment)

The disjoint two-TP-rank capture was rerun with the fixed auto-flush and NEXTN
hooks and calibrated at ranks 128/256/512 (`gsi_artifacts/rank256-512-corrected`,
`gsi_artifacts/rank512-decode-only`). 465 targets: 390 standard linears
(78 layers x 5) and 75 routed gate/up inputs. `kv_b_proj` never appears because
MLA absorbs it into `w_kc`/`w_vc`, so it is not executed as a linear.

### Spectra are not low-rank

Effective rank (entropy of the singular-value distribution) of the captured
decode activations, averaged over 78 layers x 2 TP ranks:

| family | width | eff. rank | mean rho @512 | fast frac @512, eps 0.025 |
|---|---|---|---|---|
| attn_residual (`fused_qkv_a`) | 6144 | 580 | 0.182 | 7.7% |
| attn_output (`o_proj`) | 8192 | 440 | 0.105 | 7.5% |
| q_latent (`q_b_proj`) | 2048 | 468 | 0.108 | 15.8% |
| mlp_residual (`gate_up`) | 6144 | 591 | 0.194 | 6.0% |
| shared_expert down | 1024 | 496 | 0.109 | 2.9% |
| dense_mlp down (3 layers) | 6144 | 338 | 0.027 | 57.9% |
| **routed_gate_up** | 6144 | 604 | 0.201 | **2.4%** |

These are *in-sample* numbers: the SVD was fit on the same rows. The only site
with a genuinely compressible activation manifold is the dense-MLP `down_proj`
input, which exists in 3 of 78 layers.

Extrapolating each module's residual decay (power law fit on ranks 256->512) to
each gate threshold gives the rank actually required, at the repo's `eps = 0.025`
and at the paper's `eps = 0.10`:

| family | width | mean @.025 | mean @.10 | p99 @.025 | p99 @.10 |
|---|---|---|---|---|---|
| attn_output | 8192 | 1805 | 552 | 8668 | 1426 |
| q_latent | 2048 | 2165 | 653 | 3337 | 866 |
| attn_residual | 6144 | 4904 | 1120 | 27061 | 3024 |
| mlp_residual | 6144 | 5019 | 1155 | 36899 | 3890 |
| routed_gate_up | 6144 | 5051 | 1157 | 39644 | 4042 |

At the paper's threshold `attn_output` needs k=552 for the median token — close
to the k=512 already built, which is why it is the one site that gates well. The
routed input still needs k~1157 for the median and k~4042 for p99 against a width
of 6144, so no rank both compresses and admits most tokens *there*.

### Runtime confirmation at rank 512

Decode-only functional run, 193,000 gated rows per TP rank
(`gsi_reports/rank512-decode-only-functional.tp*.json`):

- TP0 **2.04%** fast, TP1 **2.30%** fast (rank 128 was 0.31% / 0.11%).
- Per family: q_latent 5.89%, mlp_residual 2.52%, attn_residual 1.78%,
  attn_output 0.02-1.26%, mlp_intermediate 0.00-0.05%.
- Every high-fraction site is in layers 0-4 (`layers.0.o_proj` 98.4%,
  `layers.4.q_b_proj` 92.9%), i.e. hidden states still near the embedding
  manifold. Fast hits collapse to ~0 from layer 5 onward.
- Routed experts contribute no runtime rows: they have bases but no images, so
  only the calibration figure above applies to them.

### Reconciliation with the source paper (arXiv 2605.03109)

Three implementation choices here diverge from the paper and had to be corrected
before the numbers above could be judged:

1. **Threshold.** This repo defaults to `epsilon = 0.025`. The paper's headline
   operating point is **`epsilon = 0.10`** (GPT-J 6B, k=256: 99.8% fast, PPL
   ratio 0.991). Every "fast fraction" in the sections above is therefore
   measured at a gate 4x tighter than the method's own setting.
2. **Exactness.** The paper's fast path **discards the residual correction**:
   "when rho_t < eps, the correction W r_t is skipped: y_t ~= M g_t". The
   decomposition `y = (WV)(V'x) + Wr` is an identity, but the fast path drops the
   second term. The paper reports 98.6% top-1 agreement at its headline point and
   notes OPT reaches only 20% generation agreement. **Exact greedy-token equality
   was never on offer.** The rank 128 gate in this document failed a criterion
   stricter than the method claims.
3. **Speedup metric.** The paper's 15.6x is analytical, from
   `S = 1 / [f/(d/k) + (1-f)]` (Eq. 6), against a **BF16 dense** baseline; it
   concedes (S7.3) that wall-clock gains need custom kernels. No quantized
   baseline is evaluated anywhere in the paper.

### Fast fraction at the paper's threshold

Calibration fast fraction by epsilon (in-sample, 78 layers x 2 TP ranks):

| family | rank | e=0.025 | e=0.05 | e=0.10 |
|---|---|---|---|---|
| attn_output | 512 | 7.5% | 20.7% | **53.1%** |
| mlp_intermediate | 512 | 5.1% | 15.0% | 46.2% |
| q_latent | 512 | 15.8% | 22.3% | 36.9% |
| attn_residual | 512 | 7.7% | 10.8% | 20.2% |
| mlp_residual | 512 | 6.0% | 9.1% | 17.7% |
| routed_gate_up | 512 | 2.4% | 5.5% | **14.4%** |
| routed_gate_up | 256 | 0.0% | 0.0% | 0.6% |

At the paper's own threshold the fast fractions are 14-53%, not 2.4%. The
"insignificant fast fraction" finding was an artifact of the tighter gate.

### Break-even under the paper's cost model

The paper's premise is that "inference at batch size one is dominated by the cost
of reading weight matrices", so the correct accounting is **bytes read, not
MACs**. Per module, the basis `d x k` is read on every token regardless of path:

`f > 2dk / (amort * out * (bytes_W * d - bytes_img * k))`

With a W4AFP8 slow path (0.5 B/elem) rather than the paper's BF16 (2 B/elem):

| site | k | image dtype | break-even f | measured f @ e=0.10 |
|---|---|---|---|---|
| attn o_proj | 512 | fp8 | 38.1% | **53.1% — passes** |
| attn o_proj | 512 | bf16 | 44.4% | **53.1% — passes** |
| routed gate/up (x8) | 512 | fp8 | 15.0% | 14.4% — marginal |
| routed gate/up (x8) | 512 | bf16 | 18.8% | 14.4% |
| shared/dense gate_up | 512 | bf16 | 50.0% | 17.7% |
| routed gate/up (x8) | 256 | fp8 | 6.8% | 0.6% |
| routed down (x8) | 512 | any | impossible (k >= d/2) | 44.1% |

**The quantized baseline is what moves the ceiling.** The paper's compression
ratio `d/k` assumes BF16 weights. Against W4AFP8 the ratio is `d/(4k)`: at
d=6144, k=512 the *ceiling* is 3.0x, not 12x, before the always-paid basis read.
Applying the paper's own Eq. 6 to our measured fractions gives S_eff = 1.15x
(routed gate/up), 1.19x (mlp_residual), 1.99x (attn o_proj) — against BF16.
Against the W4AFP8 path actually deployed here, those are below 1.

### Memory

`gsi.footprint` for all 256 experts x 75 routed layers, per TP rank:

| rank | BF16 images | FP8 images |
|---|---|---|
| 128 | 23.4 GiB | 11.7 GiB |
| 256 | 46.9 GiB | 23.4 GiB |
| 512 | 93.8 GiB | 46.9 GiB |

Images are additive: the W4AFP8 weights must stay resident for the slow path.
With ~83 GiB of weights already loaded on an 80 GiB H100, the rank required for
a non-trivial fast fraction cannot be stored at any dtype.

## Decision

The feasibility gate is **not passed for the MoE path**, but the earlier "closed,
structurally impossible" verdict was overstated and is withdrawn. Corrected:

1. **The fidelity failure was measured against the wrong bar.** The paper's fast
   path is approximate by construction. Greedy divergence from a forced-all-slow
   control is expected behavior, not a defect. Any future gate must use the
   paper's criteria (perplexity ratio, top-1 agreement) or an explicit product
   decision to require exactness — which this method cannot provide.
2. **The fast-fraction failure was measured at the wrong threshold.** At
   `epsilon = 0.10`, k=512 gives 14-53% rather than 2.4%.
3. **What genuinely does not transfer is the quantized baseline.** GSI trades
   weight bytes for basis+image bytes. Against BF16 that is a 16x trade; against
   W4AFP8 it is 3x at k=512, and the always-paid `d x k` basis read eats most of
   what remains. This is the real reason the method does not pay here, and the
   paper cannot have seen it: it evaluates no quantized baseline (S7.4 lists
   quantization as future work, asserting orthogonality without measurement).
4. **Memory remains decisive for the routed experts.** 256 experts x 75 layers of
   images cost 46.9 GiB/rank at k=512 even in FP8, additive to ~83 GiB of
   resident weights on an 80 GiB H100. This is independent of any threshold.
5. **Direct evidence against the paper's S7.1 scaling hypothesis.** S7.1 conjectures
   that "if r_eff is bounded independently of d, the compression ratio d/k
   increases with model size". GLM-5.2 measures r_eff = 440-604 at d = 6144-8192,
   against GPT-J's k=256 sufficing at d=4096. Here r_eff grows with d, so the
   conjectured favorable scaling does not hold for this model. That is the most
   transferable result from this work.

`kernel` mode stays disabled and the fused KT routed-expert backend should not be
built for the routed path: it is memory-blocked regardless of gate tuning.

**One site is not closed.** `attn o_proj` at k=512, epsilon=0.10 clears the
batch-1 byte break-even on its own (53.1% measured vs 38.1% required with FP8
images), it is a single dense linear per layer with no expert multiplicity, and
its images cost ~1.5 GiB/rank rather than 47.

Note the inverted failure mode: the paper's OPT layer 0 fails because learned
positional embeddings spread activations across all dimensions, while here layers
0-4 are the *only* sites that gate well and everything from layer 5 collapses.

## Implementation status against the paper

Every mechanism the paper specifies is now implemented; the divergences that
produced the misjudged verdict above are closed.

| paper mechanism | status |
|---|---|
| `eps = 0.10` operating point | `calibrate.DEFAULT_EPSILON` |
| approximate fast path (`Wr` skipped) | already correct in `math.apply_gated_linear` |
| one shared basis per layer | `--basis-scope layer` (default), keyed by activation width |
| cascade init from the previous layer | `--cascade`, `subspace.cascade_basis` |
| depth-coherence diagnostic | `depth-coherence.json`, `subspace.subspace_overlap` |
| coverage of all linear maps | attention + dense/shared MLP + both routed stages |
| static-projection negative control | `GSI_GATE_OVERRIDE=always_fast` |
| Eq. 6 effective speedup | `evaluate.effective_speedup` |
| perplexity ratio / top-1 / generation agreement | `evaluate.score_run`, `--logprobs` traces |

Two things the paper does not have are implemented alongside, because the
comparison is meaningless without them: `evaluate.deployed_speedup` scores the
same run on batch-1 weight *bytes* against W4AFP8, and `evaluate.break_even_fast_fraction`
gives the fraction at which a site starts paying for itself.

Routed-expert coverage required new machinery. Those maps are consumed by a fused
grouped GEMM, so `expert_images` builds `M13_e = W13_e V` and `M2_e = W2_e U`
through the same CUTLASS kernel the slow path uses (never through a dequantized
copy), and the post-SiLU down input is captured from inside `cutlass_w4a8_moe`
via the layer id published by `runtime.routed_layer`. Following the paper, the
down basis is shared across all 256 experts at a layer rather than fitted per
expert. `GSI_EXPERT_LAYERS` restricts the build to a layer subset, which is
mandatory in practice: the full set does not fit (see Memory above).

## Live measurement, 2026-07-31 (paper-faithful configuration)

Full pipeline executed on 2xH100 NVL, TP2, W4AFP8, MTP off, top-2 substitution:
observe capture -> calibrate -> build images -> functional serve.
Artifacts: `gsi_artifacts/paper-layer/`, `gsi_reports/paper-*`.

Two implementation bugs were found only by running it, and are fixed:

- `observe_fused_intermediate` captured the whole `intermediate_q` buffer, which
  is allocated for `m*topk` rows but written only up to `expert_offsets[-1]`.
  0.1% of `routed_down` rows were uninitialized memory. The hook now slices to
  the valid row count, and `calibrate.usable_rows` drops non-finite/zero rows
  before any decomposition.
- cuSOLVER `gesvd` fails to converge on the post-SiLU SwiGLU spectra.
  `calibrate.right_singular_basis` falls back to the Gram eigendecomposition,
  then to CPU LAPACK.

### Depth coherence: the paper's S4.3 claim does not hold here

Mean cosine of principal angles between consecutive layers, rank 512
(paper: >0.90 from layer 8 onward on GPT-J):

| activation space | median | min | max | fraction >=0.90 |
|---|---|---|---|---|
| residual stream (6144) | 0.892 | 0.805 | 0.954 | 44% |
| SwiGLU intermediate (1024) | 0.637 | 0.631 | 0.643 | 0% |
| q-latent (2048) | 0.436 | 0.421 | 0.439 | 0% |
| head concat (8192) | 0.203 | 0.145 | 0.245 | 0% |

Only the residual stream is depth-coherent, and only marginally. **Cascade
initialization is unsound on GLM-5.2 outside the residual stream**; these
numbers were produced with the exact SVD, not the cascade.

### The shared per-layer basis is a net loss

Pooling the maps that read the residual stream raises effective rank from 604
(per-module) to 933, and drops the in-sample fast fraction at k=512, eps=0.10
from 14-20% per-module to **6.5%** shared. GLM-5.2's maps read different
distributions even at equal width, so the paper's one-basis-per-layer choice
costs more than the projection sharing saves.

### Runtime result

Out-of-sample, 402,000 gated rows per TP rank, k=512, eps=0.10:

| family | in-sample ff | runtime ff | break-even | deployed S |
|---|---|---|---|---|
| q_latent | 32.3% | **19.27%** | unreachable | 0.800 |
| attn_output | 30.9% | 5.24% | 44.4% | 0.795 |
| attn_residual | 6.5% | 4.22% | 117% | 0.574 |
| mlp_residual | 6.5% | 1.52% | 145% | 0.523 |
| mlp_intermediate | 5.3% | 0.77% | unreachable | 0.746 |

- **Overall fast path: 6.20% (TP0) / 6.15% (TP1).**
- **Paper's Eq. 6 aggregate: 1.066x** (against its assumed BF16 baseline).
- **Deployed speedup: 0.699x** -- a 30% slowdown against W4AFP8.
- Wall clock, eager both sides: 2.11 tok/s functional vs 3.72 tok/s exact =
  **0.57x**. Directionally confirms the byte model; not a kernel measurement.
- Fidelity: 7 of 8 prompts diverged from the exact run.

Two numbers deserve emphasis. First, **in-sample fast fractions overstate by up
to 6x**: `attn_output` calibrates at 30.9% and delivers 5.24% on fresh tokens.
Every earlier estimate in this document that came from calibration replay is an
upper bound. Second, several sites have **unreachable break-even**: at k=512 with
BF16 images, `q_latent` (d=2048) has `image_bytes*k = weight_bytes*d`, so the
image row costs exactly what the INT4 weight row costs. It has the best fast
fraction in the model, 19.27%, and still cannot pay for itself at any fraction.

### Verdict

Measured, at the paper's own operating point with its own metric: **1.07x on the
paper's formula, 0.70x deployed.** The method does not pay on this stack. The
cause is the one identified above and now confirmed end to end -- GSI trades
weight bytes for basis and image bytes, which is a winning trade against BF16 and
a losing one against packed INT4. `kernel` mode stays disabled.

Routed-expert images were not built in this run (`GSI_EXPERT_LAYERS=none`); the
~9 GiB of free VRAM after weights cannot hold them, consistent with the Memory
section above. The CUTLASS routed-image builder in `expert_images` therefore
remains unexercised on hardware.
