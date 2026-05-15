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
| 0b | Frozen-latent entropy-prior probe | Decide whether frozen DCVC-RT latents can benefit from a stronger entropy model. |
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

| Rate point | Avg bpp | Avg PSNR | Avg enc s/frame | Avg dec s/frame |
|---:|---:|---:|---:|---:|
| q0 | 0.001015 | 33.1607 | 0.00766 | 0.00930 |
| q21 | 0.003988 | 37.0718 | 0.00772 | 0.00943 |
| q42 | 0.015307 | 40.2366 | 0.00844 | 0.00965 |
| q63 | 0.054057 | 42.8234 | 0.01004 | 0.01103 |

These numbers are not yet the final BD-rate claim. They are the local baseline
curve that must be checked against official/paper numbers before ACA training.

## Final Story

DCVC-RT removes explicit optical flow, but its temporal context is still fixed
or implicit. ACA-RT upgrades that path into decoder-synchronized long-context
memory. RoPE, FlashAttention-style attention, and KV cache make the memory
mechanism practical enough to test whether adaptive temporal context lowers
content latent entropy without changing the frozen codec backbone.

