# VCT V2: Long-Context Spatiotemporal Entropy Model for Neural Video Compression

## 1. Motivation

### 1.1 What VCT Already Does Right

VCT (Mentzer et al., NeurIPS 2022) introduced a transformer-based entropy model for neural video compression. Its key insight: **spatiotemporal joint self-attention** across multiple frames in latent space.

VCT pipeline (per spatial window, batched across all patches):

```
Frame t-2 latent ──┐
Frame t-1 latent ──┤
                   ├──→ encoder_sep (3 layers, per-frame self-attn)
                   │      output: enc_{t-2}, enc_{t-1}  (B' × 64 × 768 each)
                   │
                   ├──→ concat along seq dim  →  (B' × 128 × 768)
                   │
                   ├──→ encoder_joint (2 layers, full self-attn cross frames)
                   │      output: encoded (B' × 128 × 768)
                   │
                   ├──→ decoder (5 layers, autoregressive, cross-attn to encoded)
                   │      output: (μ, σ) per token of frame t
                   │
                   └──→ GsnConditionalLocScaleShift  →  quantized latent + bits
```

The **encoder_joint** is the critical module: it performs spatiotemporal attention where every token can attend to every other token across all frames. This is architecturally correct and reminiscent of how LLMs handle long sequences.

### 1.2 The Bottleneck: GOP utilization is 6%

VCT's `encoder_joint` processes only `context_len = 2` frames (128 tokens). This is not a fundamental design limitation—it's a practical one: `context_len` was chosen at a time when training transformer models on longer sequences was computationally prohibitive.

Concretely: **LVC benchmarks (UVG, MCL-JCV, HEVC Class B/C) all use GOP=32**. VCT
attends to 2 of those 32 frames, which is **6% GOP utilization**. The remaining
29 frames of historical context the model could in principle see are simply
not given to the joint encoder.

The hypothesis of V2: **fill the GOP**. Lift `context_len` from 2 to 32 (the
industry-standard GOP length) using modern LLM building blocks. No architectural
redesign — same VCT shape, same spatiotemporal joint attention idea, just at
the sequence length the benchmark already implies.

### 1.3 Why Now

The LLM community (2023-2024) has solved exactly the problem VCT faces at scale:
- FlashAttention-2 → O(N) memory for exact attention
- RoPE → position encoding that scales to arbitrary sequence lengths
- SwiGLU + RMSNorm → better training dynamics than GELU + LayerNorm
- Mamba SSM → O(N) linear-time sequence modeling for N > 1000 tokens
- GQA / KV-cache → efficient inference with long context

V2 applies these proven primitives to the video compression domain.

---

## 2. V2 Architecture

### 2.1 Key Insight: Don't Redesign, Just Scale

The original VCT already has the right architecture—encoder_joint performs exact spatiotemporal attention. V2's only structural change is replacing this module with one that can handle longer sequences.

```
VCT V1:  encoder_joint = EncoderSection(2 layers, GELU-MLP, LayerNorm, LearnedPosition)
                              Input:  (B' ×   128 × 768)
                              Output: (B' ×   128 × 768)

VCT V2:  encoder_joint = LongCtxJointEncoder(4 layers, SwiGLU, RMSNorm, RoPE)
                              Input:  (B' × N×64 × 768)    for N up to 32 (project target)
                              Output: (B' × N×64 × 768)
```

> **Scope note (2026-05)**: project target is **N ≤ 32**; the hybrid local-global
> path described in §2.2 below is kept as future work but is NOT exercised by any
> phase in the active roadmap (§4). For N ≤ 32 the encoder degenerates to a stack
> of plain FlashAttnBlock layers operating on the full sequence.

### 2.2 LongCtxJointEncoder: Hybrid Local-Global Attention (out of scope for current roadmap)

For N ≤ 32, the entire sequence (N × 64 ≤ 2048 tokens) fits in FlashAttention-2 with full quadratic attention — this is what the project actually uses. The hybrid architecture below was originally designed for N > 32; it is preserved in the codebase but unused under the narrowed roadmap.

```
Input: seq (B' × N×64 × 768)
  │
  ├── Frame Embedding: add learned embedding per frame index
  │       frame_ids = [0]*64 + [1]*64 + ... + [N-1]*64
  │       seq = seq + self.frame_embed(frame_ids)
  │
  ├── Local Branch (FlashAttnBlock × 2):
  │       operates on last mid_window tokens (default: 16 frames = 1024 tokens)
  │       output: local_out
  │       stitched back: local_full = [seq_old[:-K], local_out]
  │
  ├── Global Branch (FlashAttnBlock × 2):
  │       operates on full N × 64 sequence
  │       output: global_out
  │
  └── Gated Fusion:
          g = σ(self.gate)          # initialized at -3 → σ ≈ 0.047
          out = g * global_out + (1-g) * local_full
```

### 2.3 Spatiotemporal Position Encoding

This is the most subtle design decision. The input sequence has two orthogonal position axes:

| Axis | Meaning | Range | Encoding Method |
|------|---------|-------|-----------------|
| Spatial | Which patch within a frame | 0..63 | 1D RoPE on token index % 64 |
| Temporal | Which frame in the GOP | 0..N-1 | Learned frame embedding |

These are **orthogonal and additive**—RoPE rotates in head-dimension space, frame embedding adds in model-dimension space. They don't interfere.

The alternative (2D RoPE) would encode both axes in the same rotation space, but requires careful frequency allocation between spatial and temporal sub-bands. The frame embedding approach is simpler, more interpretable, and easier to ablate.

### 2.4 Building Block Modernization

Every primitive in the transformer path is upgraded:

| VCT V1 | VCT V2 | Rationale |
|--------|--------|-----------|
| `nn.LayerNorm` | `RMSNorm` | Faster, equally effective (LLaMA) |
| `MLP(GELU)` (2-layer) | `SwiGLU_FFN` (gated 3-weight) | Better gradient flow (PaLM) |
| `WindowMultiHeadAttention` | FlashAttention-2 (or PyTorch SDPA) | O(N) memory, hardware-aware |
| `LearnedPosition` | RoPE (rotary) + frame embedding | Extrapolates to unseen sequence lengths |
| Look-ahead mask (decoder) | Causal flag in FlashAttn | Native support, fewer bugs |

### 2.5 Gate Initialization Strategy

The `LongCtxJointEncoder.gate` is initialized at **-3.0** (σ ≈ 0.047), meaning the global branch contributes only ~5% at training start. This is intentional:

- Early training: local branch dominates (it's the "safe" option—closer to VCT V1 behavior)
- As training proceeds: if global information proves useful, the gate will drift upward
- The gate value trajectory is itself a publishable ablation result

---

## 3. Files Changed

All modifications are in `projects/torch_vct/neural/`. No files outside this directory were touched.

### 3.1 New File: `modern_blocks.py` (371 lines)

Contains all V2 infrastructure, independent from existing code:

```
RMSNorm              — Root Mean Square Layer Normalization
SwiGLU_FFN           — SwiGLU feed-forward network
precompute_freqs_cis — RoPE frequency precomputation
apply_rotary_emb     — RoPE application to Q/K tensors
FlashAttnBlock       — Single self-attn block (RMSNorm→FlashAttn+RoPE→RMSNorm→SwiGLU)
LongCtxJointEncoder  — V2 main module (replaces encoder_joint)
MambaBlock           — Mamba SSM layer (for future N≥64 experiments)
```

### 3.2 Modified: `entropy_model_layers.py` (+175 lines at end)

Appended after the original classes:

```
TransformerBlockV2   — Modernised transformer block (RMSNorm, SwiGLU, FlashAttn cross-attn)
TransformerV2        — Stack of TransformerBlockV2 layers
```

Original classes (`LearnedPosition`, `StartSym`, `TransformerBlock`, `Transformer`, `EncoderSection`) are completely untouched.

### 3.3 Modified: `entropy_model.py` (+~40 lines, 3 targeted edits)

1. **Import**: added `TransformerV2`
2. **`__init__` signature**: added `use_v2_encoder: bool = False`, `use_v2_decoder: bool = False`
3. **`__init__` body**: conditional creation of `encoder_joint` (V1 vs V2) and `decoder` (V1 vs V2)
4. **`_get_transformer_output`**: when V2, skip external position encoding (RoPE is internal)

### 3.4 Backward Compatibility

Both flags default to `False`. The V1 path is a pure conditional branch—zero changes to the original computation graph. Running VCT with default parameters yields bit-identical output to the original codebase.

---

## 4. Experimental Roadmap

**Project scope (2026-05 revision)**: terminal target is `context_len=32`. Phases 4
and 5 from the original plan (ctx=64 / ctx=128, requiring hybrid attention and
Mamba) are deferred indefinitely. The current platform is a single RTX 6000 Ada
(48 GB VRAM); the full ctx-128 path remains out of scope, and the project
doesn't need it to make its core claim
(*long-but-tractable* context > 2 frames improves RD-rate).

Total wall-clock from Phase 0 to Phase 3b must be re-estimated after the current
6000 Ada batch-capacity benchmark.

### Phase 0: Sanity Check (current)

- `use_v2_encoder=False, use_v2_decoder=False, context_len=2`
- Verify bit-exact equivalence with original VCT
- Run 1 training step to confirm no regressions

### Phase 1: Building Block Upgrade (Experiment 1)

- `use_v2_encoder=True, use_v2_decoder=True, context_len=2`
- Same data (Vimeo) as VCT V1, only modernized primitives
- Expected: BD-rate ±2% of V1 baseline
- This validates that RMSNorm + SwiGLU + RoPE + FlashAttn don't hurt

### Phase 2: Short Context Extension (Experiment 2)

- `use_v2_encoder=True, use_v2_decoder=True, context_len=4..6`
- Vimeo septuplet upper bound is 7 frames, so 6 is the max here
- Expected: BD-rate improvement 2–5%
- First real test of whether longer context helps

### Phase 3a: Medium Context (Experiment 3, primary contribution)

- `context_len=16`, requires switching to Kinetics-400 subset
- Pure FlashAttn-2 over 1024-token sequences
- Per-device B=4, no special tricks needed
- Expected: BD-rate improvement 5–8%

### Phase 3b: Long-end Context (Experiment 4, terminal)

- `context_len=32`, Kinetics subset
- 2048-token sequences — FlashAttn-2 still handles it directly
- Per-device B=2 with `accumulate_grad_batches=4` to keep effective batch=8
- Expected: BD-rate improvement 8–12%
- **This is where the project stops.** Going beyond requires either bigger VRAM
  or the hybrid/Mamba path that §2.2 describes.

### Out of scope (originally Phase 4 / 5)

- `context_len ≥ 64`: would need to revive the hybrid local-global path in
  `LongCtxJointEncoder` or swap in `MambaBlock` for the global branch.
- KV-cache compression for inference.
- Multi-GPU FSDP / ZeRO.

These are documented in §2.2 and `modern_blocks.py` for completeness but no
experiments target them.

### Ablation Matrix

Each phase can independently toggle `use_v2_encoder` and `use_v2_decoder` to measure the contribution of encoder vs decoder modernization. Within scope:

- Gate value: trainable vs fixed (gate is a no-op for ctx ≤ 32 if hybrid is bypassed)
- RoPE theta base: 10K vs 100K (the latter for ctx=32)
- Frame embedding: present vs absent
- Effective batch via `accumulate_grad_batches`: 8 vs 16

Removed ablations (out of scope): `mid_window_frames` sweep, Global branch FlashAttn vs Mamba.

---

## 5. Design Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Position encoding | Frame emb + 1D RoPE | Simpler than 2D RoPE; interpretable; easy to ablate |
| Gate initialization | -3.0 (σ≈0.047) | Training starts with local-dominant; safe warmup |
| FlashAttention fallback | PyTorch SDPA | SDPA internally calls FlashAttn; near-zero perf loss |
| Mamba fallback | Conv1d + gating | Allows experiments without mamba-ssm installed |
| File organization | `modern_blocks.py` standalone | No pollution of original VCT code; easy to share |
| Backward compat | Boolean flags, default False | Zero-risk migration; enables A/B comparison |

---

## 6. Novelty Claims

What V2 contributes beyond VCT, stated in three precise claims:

### 6.1 Observational novelty: "GOP utilization" as a framing

Prior LVC work discusses RD performance, encoder design, entropy model
sophistication. **No one has framed "we only use 2 of the 32 frames the
benchmark assumes" as an explicit problem statement.** This re-framing of a
well-known fact (everyone uses ctx=2) into an explicit bottleneck
(*utilization = 6%*) is a viewpoint contribution. Worth 1–2 paragraphs in the
introduction.

### 6.2 Engineering + empirical novelty: making GOP-fill work

Conceptually "scale to 32 frames" is trivial. Making it train without OOMing,
diverging, or losing the gains positional encoding gave VCT requires a
specific stack:

- **FlashAttention-2** — without it, ctx=32 (seq=2048) blows up memory.
- **RoPE** — learned positions don't extrapolate; RoPE does and stays stable.
- **RMSNorm + SwiGLU** — long-sequence training stability over GELU+LayerNorm.

The main contribution is the **empirical demonstration that this specific
combination works** in the LVC entropy-model setting. This is what the bulk of
the paper defends.

### 6.3 Empirical novelty: the context-length saturation curve

The 5-step incremental experimental design (Phase 0 → 3b: ctx ∈ {2, 4, 8, 16, 32})
produces a **BD-rate vs context-length curve** that no prior LVC work has reported.
This curve tells the field:

- How marginal gains accumulate from ctx=2 to ctx=32
- Where saturation begins (if it does)
- How the slope differs by content type (UVG categories: low/high motion, scene cut)

The curve itself is a research artifact: a reference baseline for the next
generation of long-context LVC work.

---

## 7. Differentiation from Related Work

| Related work | Their angle | Our angle |
|---|---|---|
| **VCT** (NeurIPS '22) | Introduce transformer-based LVC | Extend its joint-encoder context |
| **FLAVC** (CVPR '25) | Replace MEMC with attention | Same path, but push attention length |
| **DCVC-RT** (CVPR '25) | Implicit temporal modeling for speed | They trade context for speed; we trade speed for context |
| **L-STEC** (Dec '25) | LSTM-based long-term modeling | They use LSTM; we use modern transformer |
| **LTCG** (ECCV '24) | Feature clustering for long-term retrieval | They search clusters; we use dense attention |

The differentiator is consistent: **dense spatiotemporal attention over the full
GOP**, achieved via modern primitives. Each related work either trades context
away (DCVC-RT), uses a different mechanism (LSTM, clustering), or operates at a
shorter context (VCT, FLAVC). None fill the GOP with attention.

---

## 8. Key References

- **VCT**: Mentzer et al., "VCT: A Video Compression Transformer", NeurIPS 2022
- **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism", 2023
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- **LLaMA**: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **HiFiC**: Mentzer et al., "High-Fidelity Generative Image Compression", NeurIPS 2020
