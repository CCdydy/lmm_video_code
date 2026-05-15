# VCT V2 Architecture — One-page Reference

> 2026-05 修订版。详细推导见 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)，
> 数据集见 [`DATASETS.md`](DATASETS.md)，env 与运行命令见 [`README.md`](README.md)。

## 0. 一句话定位

> "VCT 已经在做 spatiotemporal joint attention，但只用了 GOP 的 6%（2/32 帧）。
> 我们用 FlashAttention-2 + RoPE + RMSNorm + SwiGLU 把它填满整个 GOP。"

Main contribution metric: **GOP utilization 6% → 100%**。

---

## 1. Pipeline（V1 / V2 共用骨架）

```
                 Frame_{t-N+1} ... Frame_t          (input: N 帧, N=2/4/8/16/32)
                            ↓ per-frame
        ┌────────────────  ELIC Analysis  ────────────────┐
        │   (transforms.py: ELICAnalysis, 共用 V1/V2)      │
        └────────────────────────────────────────────────┘
                            ↓
              latent y ∈ R^(B, N, 192, 16, 16)
                            ↓
        ┌────────────────   Patcher    ───────────────────┐
        │  window=8 → 4 patches/frame, 64 tokens/patch    │
        │  window=4 (decoder side) → 16 patches/frame     │
        └────────────────────────────────────────────────┘
                            ↓
          tokens ∈ R^(B'=B×4, 64, 768)   per frame
                            ↓
       ┌─── encoder_sep (3 layers, per-frame self-attn) ───┐
       │   (entropy_model_layers.py: 共用 V1/V2)            │
       └────────────────────────────────────────────────────┘
                            ↓
           concat across N frames → (B', N×64, 768)
                            ↓
       ╔═════════════ encoder_joint ═══════════════════════╗
       ║  V1: 2 layers, LayerNorm + GELU + LearnedPos      ║   if use_v2_encoder=False
       ║  V2: 4 layers, RMSNorm + SwiGLU + RoPE + FA-2     ║   if use_v2_encoder=True
       ║         (LongCtxJointEncoder, modern_blocks.py)   ║
       ╚════════════════════════════════════════════════════╝
                            ↓
                  encoded ∈ R^(B', N×64, 768)
                            ↓
       ╔═════════════════ decoder ═════════════════════════╗
       ║  V1: 5 layers, autoregressive cross-attn          ║   if use_v2_decoder=False
       ║  V2: 5 layers, TransformerV2 (modern primitives)  ║   if use_v2_decoder=True
       ╚════════════════════════════════════════════════════╝
                            ↓
           (μ, σ) per current-frame token → GsnConditional → bits
                            ↓
              ELIC Synthesis → reconstructed Frame_t
```

**唯二 V2 替换点**：`encoder_joint` 和 `decoder`。其他组件（ELIC、Patcher、
encoder_sep、bottlenecks）V1 / V2 完全共用，**保证 V1 路径 bit-exact**。

---

## 2. V2 Building Blocks（`neural/modern_blocks.py`）

| 类 | 用途 | Phase 0–3b 触发？ |
|---|---|---|
| `RMSNorm` | 替换所有 LayerNorm | ✅ |
| `SwiGLU_FFN` | 替换 2-layer GELU MLP | ✅ |
| `precompute_freqs_cis` + `apply_rotary_emb` | RoPE 位置编码 | ✅ |
| `FlashAttnBlock` | RMSNorm → FA-2 + RoPE → RMSNorm → SwiGLU | ✅ V2 核心 |
| `LongCtxJointEncoder`（**non-hybrid 分支**） | 一栈 FlashAttnBlock + frame embedding | ✅ ctx ≤ 32 走这条 |
| `LongCtxJointEncoder` hybrid 分支（local + global + gate fusion） | 设计用于 N > 32 | ❌ 不触发，**代码保留** |
| `MambaBlock` | 设计用于 ctx=128 替换 global 分支 | ❌ 不触发，**代码保留** |

---

## 3. 实验矩阵（5 步 incremental）

| # | 名称 | use_v2_* | ctx | 数据 | 期望 BD-rate | 主机 | 状态 |
|---|---|---|---|---|---|---|---|
| 0 | V1 复现 | F/F | 2 | Vimeo 全集 | 对得上 paper | 6000 Ada | ⏳ 等数据 |
| 1a | V2 building blocks | T/T | 2 | Vimeo | ±2% | 6000 Ada | 等 P0 |
| 1b | V2 + DDP | T/T | 2 | Vimeo | 同 1a | 6000 Ada (多卡) | 🟡 NCCL/complex 测试需多 GPU 环境 |
| 2 | short context | T/T | 4–6 | Vimeo | −3% to −6% | 6000 Ada | 等 P1 |
| 3a | medium context | T/T | 16 | Kinetics 5% subset | −5% to −10% | 6000 Ada | 等 P2 |
| 3b | **GOP-fill (terminal)** | T/T | **32** | Kinetics 5% | **−8% to −12%** ← main result | 6000 Ada | 等 P3a |
| 3c | ablation: + ALiBi | T/T + ALiBi | 32 | Kinetics 5% | +0–2% on top, plus figure | 6000 Ada | **deferred** |

**OUT OF SCOPE**：ctx ≥ 64、Mamba 训练路径、hybrid local-global gated fusion、KV-cache 压缩、MCL-JCV 评测。

---

## 4. 双机分工

| | **dev box** RTX 5090 Laptop | **training box** RTX 6000 Ada |
|---|---|---|
| sm | 120 (Blackwell) | 89 (Ada Lovelace) |
| VRAM | 24 GB | 48 GB |
| 用途 | sanity / wiring / 单 step 调试 / 数据准备脚本 | 所有真实训练、ablation、长跑 |
| FA-2 通路 | PyTorch SDPA flash backend (sm_120 已验证) | `flash_attn` pip 包（成熟 sm_89 wheel） |
| ALiBi (Phase 3c) | 用 SDPA + attn_mask（慢但能跑小测） | 用 `flash_attn` alibi_slopes（in-kernel） |
| env 状态 | ✅ 已重建并验证（torch 2.7.1+cu128） | ⏳ 待重建（git pull 后按 README 走） |
| 数据集 | mini 子集（仅小测） | Vimeo / Kinetics 5% / UVG（**完整**） |

代码通过 `github.com/CCdydy/lmm_video_code` 同步。

---

## 5. Phase 推进所需的硬阻塞 / 软阻塞

🔴 **硬阻塞**：
1. **数据集** — Vimeo 拷贝中。Phase 0 不需要 Kinetics，先 Vimeo 就够。
2. **6000 Ada env 重建** — git pull + 按 README "Environment" 装。

🟡 **软阻塞**：
1. Phase 1b 的 NCCL/complex buffer bug 在 torch 2.7 + NCCL 2.26 上是否仍存在 —
   **只能在 6000 Ada 多 GPU 环境上确认**，单卡 5090 上 2 进程 NCCL init 不稳。
2. 如 bug 仍在 → 按已知方案 patch `modern_blocks.py`（拆 `(real, imag)` 双 float32 buffer）。
   如 bug 修了 → 无动作。

⚪ **不阻塞**：
- dev box env 完整，V1/V2 forward pass 都通过 GPU smoke test
- PyTorch SDPA flash backend 在 sm_120 上 16.5 ms / 2048 seq（已基准）
- functional_tensor shim 在 sitecustomize.py 里就位
- 文档（README / Design Log / DATASETS）都收敛到 ctx ≤ 32 范围

---

## 6. 数据可用后的下一步

1. 确认 Vimeo 路径结构（`sep_trainlist.txt` / `sequences/<group>/<seq>/im{1..7}.png`）
2. 改 [vimeo.yaml](projects/torch_vct/config/datamodule/vimeo.yaml) 的 `data_dir`
3. 本机跑 Phase 0 `fast_dev_run`（1 step 端到端，B=1，只验 wiring）
4. 跑通后 commit
5. 6000 Ada env 建起来 → 同步数据 + 代码 → 开始 Phase 0 真长跑
