# ACA-RT Project Plan

## Project Identity

| Field | Value |
|---|---|
| Working name | ACA-RT: Adaptive Context Aggregation for Real-Time Neural Video Compression |
| Backup name | LCA-NVC, if the final real-time claim does not hold |
| Backbone | DCVC-RT (Microsoft, CVPR 2025), frozen plug-in style |
| One-line paper positioning | Preserve DCVC-RT's implicit single-state propagation while giving the entropy model explicit lossless access to past-K decoded features, so it can choose temporal scope per latent position. |
| DCVC-family narrative | DCVC (2021) through DCVC-FM (2024) rely on explicit motion; DCVC-RT (2025) removes motion and uses implicit single-state propagation; ACA-RT does not revive motion, but expands that implicit state into explicit multi-scope memory. |
| Time budget | 8 months, counted from 2026-05-15 |

Final project direction:

> In a frozen DCVC-RT backbone, use LLM-style long-context memory to augment the
> temporal entropy prior, so each latent position can adaptively choose whether
> to rely on DCVC-RT's original context or on explicit last-2 / last-8 / last-32
> decoded-feature memory when coding the content latent.

The target is lower entropy-coding bitrate for `y` while preserving DCVC-RT's
analysis transform, synthesis transform, quantization path, decoder, and
bitstream structure.

## Invariants

These constraints are no longer open for debate:

| Invariant | Meaning |
|---|---|
| DCVC-RT backbone | DCVC-RT is the backbone. FLAVC remains only a stop-loss escape hatch. |
| Frozen-codec semantics | Bulk codec parameters are not retrained. |
| `y_hat` preservation | ACA changes only `(mu, sigma)` / entropy parameters, not `y_hat`. |
| Encoder/decoder symmetry | `res_prior_param_decoder` behavior must remain consistent on compress and decompress paths. |
| Online replay training | Frozen DCVC-RT forward produces ACA inputs online; do not dump full Vimeo tensors because that would be 8-16 TB. |
| Gate init biased to DCVC-RT | The safe failure mode is "no improvement", not "regression". |
| Exp 0a gate | DCVC-RT UVG inference reproduction is the Step 0 gate. |

## Pivot 4 Locked Architecture

The locked design is 4-way spatial routing per latent/context position at
`H/8 x W/8`. There is no `alpha_null`: S0 already represents minimum useful
context via DCVC-RT's original one-frame implicit state.

| Mode | Scope | Implementation |
|---|---|---|
| S0 | DCVC-RT original 1-frame implicit context | Reuse `temporal_prior_encoder(ctx_t)`. |
| S1 | K=2 sliding window | Cross-attention from `ctx_t` to DPB features `t-1..t-2`. |
| S2 | K=8 sliding window | Same cross-attention path with K=8. |
| S3 | K=32 full memory | K=32 cross-attention with RoPE over temporal-distance dimension and FlashAttention-2. |

Fusion:

```text
C_adapt = sum_k alpha_k * C_k
all C_k have shape (B, 256, H/8, W/8)
```

The only default injection point is:

```text
temporal_prior_encoder(ctx_t) → temporal_prior_encoder(C_adapt)
```

Expected new parameters: about 850K-1M.

Implementation route:

```text
Option A: Minimal swap, default.
Option B: Full refinement head, only if Option A cannot produce evidence.
```

```text
DCVC-RT backbone frozen
    ↓
keep y / z latents, quantization, decoder, bitstream structure
    ↓
new DPB memory: last 32 decoded latent/context features
    ↓
history features → K/V projections + RoPE + KV cache
current-frame query feature → Q projection
    ↓
multi-scope cross-frame attention:
    C0  = original RT context
    C2  = attend last 2 frames
    C8  = attend last 8 frames
    C32 = attend last 32 frames
    ↓
per-token scope gate:
    C_adapt = α0 C0 + α2 C2 + α8 C8 + α32 C32
    ↓
z_hat + C_adapt → means/scales
    ↓
entropy coding y
```

## Core Contributions

| ID | Contribution | Concrete claim |
|---|---|---|
| C1 | Per-latent-position temporal scope routing on a frozen LVC backbone | Method contribution. |
| C2 | RD-complexity cost regularization | Add `beta * sum(alpha_k * c_k)` so the gate uses long context only when its rate benefit exceeds cost. |
| C3 | KV-Cached Temporal Memory | Real KV cache across video frames: K/V projections are reused across frames, not recomputed. |

## Technical Landmines

| # | Risk | Required handling |
|---:|---|---|
| 1 | `q_dec` leak | `params_aca = cat([params_base[:, :128].detach(), params_aca_raw[:, 128:]], dim=1)`. `q_dec` always comes detached from baseline. |
| 2 | `CUSTOMIZED_CUDA_INFERENCE` namespace pollution | Probe startup must patch `cuda_inference`, `layers`, `video_model`, and `image_model` namespaces to `False`; otherwise custom kernels break autograd. |
| 3 | Mix-coefficient dead initialization | Use `C_adapt = C_rt + gamma * DeltaC`, `gamma_init = 0.01` not 0, and zero-init attention `out_proj`. |
| 4 | High-LR training spike | Use warmup + cosine decay, held-out early stop, grad clip 0.5-1.0, and optional mixed precision only after fp32 sanity. |
| 5 | Single-sequence overfit | Must use multi-sequence + multi-seed; seed variance above 5 percentage points means unstable. |

## Frozen-backbone Training Strategy

Do not train DCVC-RT from scratch. Do not retrain the encoder or decoder.

First dump tensors from pretrained DCVC-RT:

```text
y_hat
z_hat
original context
previous decoded latent/context features
```

Then train only the added ACA entropy prior:

```text
loss = NLL(y_hat) + scope cost regularization
```

This isolates the research question: whether a stronger decoder-synchronized
temporal entropy prior can reduce content-latent rate for the same frozen
codec backbone.

## Experiment Order

| Step | Experiment | Purpose |
|---|---|---|
| 0a | DCVC-RT official inference on UVG_1080p | Establish official baseline, bitstreams, and environment sanity. |
| 0b | Single-sequence frozen-latent entropy-prior probe | Early signal check only; not sufficient for architecture lock. |
| 0b+ | Multi-sequence frozen-latent entropy-prior probe | Verify content generalization before Step 1. |
| 1 | Fixed last-2 / last-8 / last-32 context | Measure context-length sensitivity without adaptive gating. |
| 2 | Multi-scope attention without gate | Test whether combined context scopes help before adding token routing. |
| 3 | ACA per-token gate | Let each token choose C0/C2/C8/C32 adaptively. |
| 4 | Scope cost regularization | Penalize expensive long context when it does not pay for itself in NLL. |
| 5 | Level-2 bitstream evaluation | Re-encode with ACA entropy params and compare bitstream sizes against official `.bin` baselines for matched `y_hat`. |
| 6 | KV-cache speed test | Measure actual speed/memory tradeoff on RTX 6000 Ada. |

## Exp 0a Gate

Only one experiment is currently considered active for the baseline track:

```text
Exp 0a = DCVC-RT official inference on UVG_1080p
```

It has four concrete jobs:

1. **Step 0 gate**: verify inference path, weights, environment, and UVG data format before touching ACA.
2. **Baseline BD-rate numbers**: produce four UVG rate points `(bpp, PSNR)` for later ACA comparisons.
3. **Paper consistency check**: compare against DCVC-RT paper Fig. 1; if the gap is above tolerance, debug environment/data before ACA.
4. **Official bitstreams**: preserve `out_bin/UVG/*.bin` as the baseline bitstreams for later Level-2 re-encode comparisons.

It does not validate ACA design, frozen-latent entropy gains, or KV-cache speed.
Those are Step 0b and later experiments.

Observed local baseline outputs:

| Item | Value |
|---|---|
| Output dir | `/home/zzy/Desktop/DCVC/out_bin/UVG` |
| Sequences | 7 UVG 1080p sequences |
| Rate points | `q0`, `q21`, `q42`, `q63` |
| Expected jobs | 28 sequence/rate pairs |
| Observed files | 28 `.bin` + 28 `.json` |

Current quick aggregate over the 28 JSON files:

| Rate point | QP | Avg bpp | Avg PSNR YUV | Avg PSNR Y | Avg enc ms/frame | Avg dec ms/frame |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0010 | 33.16 | 31.25 | 7.66 | 9.30 |
| 1 | 21 | 0.0040 | 37.07 | 35.51 | 7.72 | 9.42 |
| 2 | 42 | 0.0153 | 40.24 | 39.08 | 8.44 | 9.65 |
| 3 | 63 | 0.0541 | 42.82 | 42.07 | 10.04 | 11.03 |

Overall speed on RTX 6000 Ada is about 118.1 fps encode / 101.5 fps decode.
The paper reports 125.2 / 112.8 fps on A100, so the local speed is within a
reasonable hardware/software gap. The RD curve is monotonic, spans about 9.7 dB,
and per-sequence bpp differences match expected content difficulty. Step 0 is
therefore considered passed within the locally verifiable scope. Exact BD-rate
alignment against paper Fig. 1/2 is still the final sanity check before making
paper-level claims.

## Step 0b Status

Current verdict:

```text
Step 0b: not passed yet
Reason:
    within-sequence signal exists
    attention is unstable
    cross-sequence generalization fails
Next:
    run multi-sequence Step 0b+ before architecture lock
```

The single-sequence probe is useful as an early warning, but it is not a pass
criterion. Bosphorus can improve, which means historical context contains signal
for predicting `y`. HoneyBee regresses heavily under a Bosphorus-trained
correction, which means the learned module is currently sequence-specific rather
than a general temporal entropy prior. The current failure mode is not "context
has no signal"; it is "the ACA module has not learned when to leave a good
DCVC-RT prior unchanged."

Attention is not stable enough to be the locked mainline yet. Re-running the
same attention probe with different random initialization can swing validation
from a large gain to nearly flat. Mean-pool memory is a strong fallback because
it is deterministic and has shown same-sequence gains, but it is not yet a
replacement for Level C. It shows that pooled temporal memory carries signal and
should be included in the expanded probe.

## Step 0b+ Multi-sequence Probe

Decision:

```text
Do not enter Step 1 yet.
Do not pivot to Pivot 3 yet.
Run multi-sequence Step 0b+.
```

Minimum sequence coverage:

```text
train mix:
    Bosphorus
    HoneyBee
    Beauty
    ReadySteadyGo

held-out:
    ShakeNDry
```

This train/held-out split is intentionally content-diverse:

| Sequence | Role | Coverage |
|---|---|---|
| Bosphorus | train | slow pan, static sea, high-bpp behavior |
| HoneyBee | train | near-static content, micro vibration, low-bpp behavior |
| Beauty | train | face, slow motion |
| ReadySteadyGo | train | fast motion, camera shake, complex motion |
| ShakeNDry | held-out | spray / high-frequency texture, true OOD relative to the train mix |

Use ShakeNDry rather than YachtRide for holdout because YachtRide is too similar
to Bosphorus; an overfit prior could appear to generalize there without really
handling a different content type.

Run four models side by side:

| ID | Model | Purpose |
|---|---|---|
| M0 | identity / no change | Sanity baseline; must exactly preserve DCVC-RT prior. |
| M1 | mean-pool K=8 | Deterministic strong baseline for pooled memory. |
| M2 | naive attention K=8 | Current attention implementation, measured honestly. |
| M3 | stabilized attention K=8 | Candidate mainline attention with residual-safe initialization. |

Run all four models on the same train/eval split, seed pool, schedule, and NLL
measurement. Only the memory module changes.

Stabilized attention requirements:

```text
C_adapt = C_rt + gamma * DeltaC

gamma init: 0.01
attention out_proj: zero-init
scope gate logits: RT slot bias +2, others 0
    softmax init ≈ alpha_rt 0.88, others 0.04 each
normalization: RMSNorm before qkv
optimizer: qkv projection LR 1e-4, other new layers LR 1e-3
schedule: 2k-step warmup, then cosine decay
training: gradient clipping at 1.0
reporting: seeds 42, 123, 7; report mean and std
```

The identity path must be safe at initialization. ACA should not begin training
by applying a large context rewrite such as a 0.79-strength correction to a
low-bpp sequence whose original prior is already accurate.

Two additional diagnostics are mandatory:

1. **Variance-reduction baseline**: run stabilized M3 once with the attention
   q/k/v path random-initialized but frozen. If frozen random attention also
   performs well, the learned q/k/v representation is not the source of the gain;
   the benefit is likely routing or residual mixing.
2. **Gate-activity check**: each validation epoch logs `mean(alpha_rt)` and
   `entropy(alpha)`. If stabilized M3 ends with `mean(alpha_rt) < 0.3`, treat it
   as suspicious even if held-out bpp passes, because the model is still rewriting
   the original context too aggressively.

Operational decisions for the next run:

| Item | Decision |
|---|---|
| Training steps | 3000 steps, i.e. 2x the 1500-step single-sequence probe. |
| Held-out eval cadence | Run ShakeNDry validation every 1000 steps and at final. |
| Seeds / hardware | Run 4 models × 3 seeds sequentially on the RTX 6000 Ada unless extra GPUs are explicitly available. |
| Loss | NLL-only on fixed `y_hat` / estimated bits; no reconstruction loss. Scope cost is reserved for later gated runs after 0b+ passes basic generalization. |

Pass criteria:

```text
PASS:
    mixed-train validation NLL / estimated bpp decreases
    held-out frames decrease
    held-out sequence delta <= 0, or at worst mild regression < +2%
    attention seed variance is controlled
    learned gate does not always choose strong context modification

FAIL:
    any model has held-out ShakeNDry delta > +5%
    held-out sequence remains > +10% worse
    or attention seed variance remains > 5 percentage points
```

Interpretation tree:

```text
Result A:
    attention generalizes across sequences and seeds are stable
    → continue Level C as planned

Result B:
    mean-pool generalizes, attention remains unstable
    → Level C mainline becomes pooled memory base + lightweight gate.
      Attention remains an optional refinement module, and the first paper
      version does not depend on attention.

Result C:
    both mean-pool and attention fail cross-sequence validation
    → abandon this ACA prior form and revisit Pivot 3 / conservative context bank
```

## Final Story

DCVC-RT removes explicit optical flow, but its temporal context is still fixed
or implicit. ACA-RT upgrades that path into decoder-synchronized long-context
memory. RoPE, FlashAttention-style attention, and KV cache make the memory
mechanism practical enough to test whether adaptive temporal context lowers
content latent entropy without changing the frozen codec backbone.
