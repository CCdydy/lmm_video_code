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

### 1.2 The Bottleneck

VCT's `encoder_joint` processes only `context_len = 2` frames (128 tokens). This is not a fundamental design limitation—it's a practical one: `context_len` was chosen at a time when training transformer models on longer sequences was computationally prohibitive.

The hypothesis of V2: **lifting this context window from 2 frames to N frames (up to 128) using modern LLM building blocks will yield significant RD-performance gains** without changing the architectural philosophy.

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
                              Input:  (B' × N×64 × 768)    for N up to 128
                              Output: (B' × N×64 × 768)
```

### 2.2 LongCtxJointEncoder: Hybrid Local-Global Attention

For N ≤ 32, the entire sequence (N × 64 ≤ 2048 tokens) fits in FlashAttention-2 with full quadratic attention. For N > 32, we employ a hybrid architecture:

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

### Phase 0: Sanity Check (current)

- `use_v2_encoder=False, use_v2_decoder=False, context_len=2`
- Verify bit-exact equivalence with original VCT
- Run 1 training step to confirm no regressions

### Phase 1: Building Block Upgrade (Experiment 1)

- `use_v2_encoder=True, use_v2_decoder=True, context_len=2`
- Same data as VCT V1, only modernized primitives
- Expected: BD-rate ±2% of V1 baseline
- This validates that RMSNorm + SwiGLU + RoPE + FlashAttn don't hurt

### Phase 2: Short Context Extension (Experiment 2)

- `use_v2_encoder=True, use_v2_decoder=True, context_len=4`
- Requires: minor dataloader change to serve 4-frame groups
- Expected: BD-rate improvement 2-5%
- First real test of whether longer context helps

### Phase 3: Medium Context (Experiment 3)

- `context_len=8` to `context_len=32`
- Full FlashAttention-2 can still handle this (≤2048 tokens)
- Expected: BD-rate improvement 5-10%

### Phase 4: Long Context (Experiment 4)

- `context_len=64`
- Hybrid local+global branches become active
- Expected: BD-rate improvement 8-12%

### Phase 5: Very Long Context (Experiment 5)

- `context_len=128`
- Add KV-cache compression, swap global branch to Mamba
- Expected: BD-rate improvement 10-15%

### Ablation Matrix

Each phase can independently toggle `use_v2_encoder` and `use_v2_decoder` to measure the contribution of encoder vs decoder modernization. Additionally:

- Gate value: trainable vs fixed
- `mid_window_frames`: 4, 8, 16, 32
- Global branch: FlashAttn vs Mamba (for N≥64)
- RoPE theta base: 10K vs 100K (for longer sequences)
- Frame embedding: present vs absent

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

## 6. Key References

- **VCT**: Mentzer et al., "VCT: A Video Compression Transformer", NeurIPS 2022
- **FlashAttention-2**: Dao, "FlashAttention-2: Faster Attention with Better Parallelism", 2023
- **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- **LLaMA**: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer", 2020
- **RMSNorm**: Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **HiFiC**: Mentzer et al., "High-Fidelity Generative Image Compression", NeurIPS 2020
