# Level C: LLM-style ACA-RT

Final project direction:

> In a frozen DCVC-RT backbone, replace or augment the original temporal entropy
> prior with LLM-style long-context mechanisms, so each latent token can
> adaptively choose whether to use recent 2-frame, 8-frame, or 32-frame history
> when coding the content latent.

The target is lower entropy-coding bitrate for `y` while keeping the DCVC-RT
analysis transform, synthesis transform, quantization path, decoder, and
bitstream structure synchronized with the official codec.

## Architecture

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
| C1 | Adaptive Context Attention | Each latent token adaptively selects temporal context length instead of using one fixed history window. |
| C2 | Multi-scope Temporal Memory Prior | DCVC-RT's short/implicit temporal context becomes a decoder-synchronized memory over last-2 / last-8 / last-32 frames. |
| C3 | KV-Cached Efficient Context Modeling | Historical K/V tensors are computed once and cached, avoiding repeated long-context attention work for every frame. |

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
