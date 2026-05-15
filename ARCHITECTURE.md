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
| 0 | V1 复现 | F/F | 2 | Vimeo 全集 (~82 GB) | 对得上 paper | 6000 Ada | `fast_dev_run` 通过 |
| 1a | V2 building blocks | T/T | 2 | Vimeo | ±2% | 6000 Ada | `fast_dev_run` 通过 |
| 1b | V2 + DDP | T/T | 2 | Vimeo | 同 1a | 6000 Ada (多卡) | 🟡 NCCL/complex 测试需多 GPU 环境 |
| 2 | short context | T/T | 4–6 | Vimeo | −3% to −6% | 6000 Ada | 等 P1 |
| 3a | medium context | T/T | 16 | Kinetics 5% subset | −5% to −10% | 6000 Ada | 等 P2 |
| 3b | **GOP-fill (terminal)** | T/T | **32** | Kinetics 5% | **−8% to −12%** ← main result | 6000 Ada | 等 P3a |
| 3c | ablation: + ALiBi | T/T + ALiBi | 32 | Kinetics 5% | +0–2% on top, plus figure | 6000 Ada | **deferred** |

**OUT OF SCOPE**：ctx ≥ 64、Mamba 训练路径、hybrid local-global gated fusion、KV-cache 压缩、MCL-JCV 评测。

---

## 4. 当前平台

当前 checkout 已在 RTX 6000 Ada 主机上，不再按“5090 dev box + 6000 Ada training box”
分工推进。旧 5090 batch 数据只作为历史参考。

| Item | Value |
|---|---|
| Host | `ZZY-Desktop` |
| GPU | RTX 6000 Ada, sm_89, 48 GB |
| Repo | `/home/zzy/Desktop/NeuralCompression` |
| Training env | `/home/zzy/anaconda3/envs/torch_vct` |
| Data | Vimeo / Kinetics / UVG raw 均在本机可访问 |

完整平台记录见 [`PLATFORM.md`](PLATFORM.md)。

---

## 5. Phase 推进所需的硬阻塞 / 软阻塞

🔴 **硬阻塞**：
1. 决定 Phase 0 是否做短训或直接跳到 Phase 1 长训。
2. Phase 1a 长训前用更多 benchmark steps 复测稳定吞吐。

🟡 **软阻塞**：
1. 当前机器是单卡 RTX 6000 Ada，Phase 1b 的 DDP/NCCL 只能在多 GPU 环境上验证。
2. RoPE complex buffer 已拆成 `(real, imag)` 双 float32 buffer；单卡 DDP setup 已通过，
   多卡 DDP 仍需复验。

⚪ **不阻塞**：
- Vimeo 数据已就绪 `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet`
- Kinetics-400 完整目录已在 `/media/zzy/data/kinetics-dataset/k400`
- UVG raw 已在 `/media/zzy/mydata/UVG` 和 `/media/zzy/mydata/UVG_720p`
- 当前 6000 Ada 已通过 V1 与 V2 `fast_dev_run`
- 文档（README / PLATFORM / DATASETS）已收敛到当前 6000 Ada 平台

---

## 7. 已知问题 / Known issues

### Patcher 600 MB identity-kernel 分配（上游 VCT 遗留）✅ 已修复 (commit aa47ca1)

`neural/patcher.py:_window_partition_conv2d` 在 `patch_size ≠ stride`（encoder 路径）
时原本构造 `torch.diag(ones(C × patch_size²))` = (12288 × 12288) identity kernel ≈
**600 MB**，conv2d 期间不释放。已替换为两次 `Tensor.unfold`（纯 view, 零分配）。

数值等价性由 [`tests/neural/test_patcher_unfold_equivalence.py`](projects/torch_vct/tests/neural/test_patcher_unfold_equivalence.py)
守护：CPU 上 forward bit-exact，CUDA 上 TF32 关闭后 forward bit-exact，backward
gradient 差 < 1 fp32 ulp。

## 8. batch capacity

复现脚本：[`scripts/benchmark_batch_capacity.py`](projects/torch_vct/scripts/benchmark_batch_capacity.py)，
当前 RTX 6000 Ada quick sweep（bf16 AMP, `--steps 1`, Vimeo 7-frame septuplet）：

| config | B | alloc_GB | reserved_GB | sec/step | samples/sec |
|---|---:|---:|---:|---:|---:|
| V1 ctx=2 | 1 | 10.15 | 10.32 | 0.507 | 1.97 |
| V1 ctx=2 | 2 | 18.04 | 18.65 | 0.567 | 3.53 |
| V1 ctx=2 | 4 | 33.69 | 34.15 | 0.825 | 4.85 |
| V1 ctx=2 | 5 | 41.58 | 43.46 | 1.053 | 4.75 |
| V1 ctx=2 | 6 | OOM | — | — | — |
| V2 ctx=2 | 1 | 11.41 | 11.59 | 0.587 | 1.70 |
| V2 ctx=2 | 2 | 20.48 | 20.80 | 0.600 | 3.34 |
| V2 ctx=2 | 4 | 38.46 | 38.96 | 0.861 | 4.65 |
| V2 ctx=2 | 5 | OOM | — | — | — |

实用上限：V1 B=5，V2 B=4。长训前建议 `--steps 3` 或更高再复测一次，减少单步噪声。

旧 RTX 5090 Laptop 历史参考值：

historical dev box (RTX 5090 Laptop, 24 GB, bf16 AMP, Vimeo 7-frame septuplet):

| config | B | alloc_GB | sec/step | samples/sec |
|---|---|---|---|---|
| V1 ctx=2 | 1 | 10.17 | 0.340 | 2.95 |
| V1 ctx=2 | **2** | **18.01** | **0.573** | **3.49** ← 实际上限 |
| V1 ctx=2 | 4 | OOM | — | — |
| V2 ctx=2 | 1 | 10.53 | 0.335 | 2.99 |
| V2 ctx=2 | 2 | 18.70 | 0.593 | 3.37 |
| V2 ctx=2 | 4 | OOM | — | — |

在当前 6000 Ada 上，V2 ctx=2 比 V1 ctx=2 多约 4.8 GB @ B=4，主要上限仍来自
ELIC per-frame 激活。

## 9. 待决策：Phase 0 V1 复现的范围

旧 5090 dev box 上跑完整 1M-step V1 复现需要 26 天；当前 6000 Ada 吞吐更高，但
完整 Phase 0 仍然会挤占 V2 实验时间。候选方案仍然是：

| 选项 | 内容 | 时间 | 优势 | 风险 |
|---|---|---|---|---|
| **A** | **跳过 Phase 0 完整复现**，直接 Phase 1 (V2 ctx=2)，V1 baseline 用 paper 公布的 BD-rate 数字 | 0 (省略) | 最高效；马上能拿 V2 数据 | 论文需说明对比方法 |
| **B** | Phase 0 短训练 (~200K 步)，拿 V1 早期收敛曲线作内部对比 | 约 2.5 天 quick-estimate | 内部一致；不依赖外部数字 | 不是 paper 终值，只能做相对比较 |
| **C** | Phase 0 完整复现 1M 步 | 约 12 天 quick-estimate | 最严谨 | 机会成本高 |

agent 推荐 **选项 A**，理由：

1. VCT paper V1 的 BD-rate 在 UVG / MCL-JCV 上已是公开 baseline。审稿人不会
   要求自己复现，写明"V1 数字取自 [VCT, NeurIPS'22]"即可。
2. 完整复现一个已有结果的机会成本高，尤其会挤占 V2 sub-experiment。
3. 选项 B 的"短训练曲线对比"本质是 noise—— V1 / V2 短训练阶段都没收敛，曲线
   形状的差异可能更多反映初始化随机性而非架构差异。

**决策悬而未定**。这台机器在等 user 拍板。

---

## 6. 下一步

1. 决定 Phase 0 采用 A/B/C 哪个复现范围。
2. Phase 1a 长训：V2 enc+dec, `context_len=2`, single-GPU non-DDP。
3. Phase 2/3：先 Vimeo `context_len=4..6`，再 Kinetics 5% 子集 `context_len=16/32`。
