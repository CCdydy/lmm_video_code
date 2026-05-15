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
| 0 | V1 复现 | F/F | 2 | Vimeo 全集 (~82 GB) | 对得上 paper | 6000 Ada | ✅ fast_dev_run @ dev box B=1 通过；真长跑等 6000 Ada |
| 1a | V2 building blocks | T/T | 2 | Vimeo | ±2% | 6000 Ada | 等 P0 长跑 |
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
- **Phase 0 fast_dev_run 端到端通过** @ dev box (V1, ctx=2, B=1, Vimeo)
- Vimeo 数据已就绪 `/home/zzy/data/vimeo_septuplet/`（91701 段 × 7 PNG = 641907，完整）
- PyTorch SDPA flash backend 在 sm_120 上 16.5 ms / 2048 seq（已基准）
- functional_tensor shim 在 sitecustomize.py 里就位
- 文档（README / Design Log / DATASETS）都收敛到 ctx ≤ 32 范围

---

## 7. 已知问题 / Known issues

### Patcher 600 MB identity-kernel 分配（上游 VCT 遗留）✅ 已修复 (commit aa47ca1)

`neural/patcher.py:_window_partition_conv2d` 在 `patch_size ≠ stride`（encoder 路径）
时原本构造 `torch.diag(ones(C × patch_size²))` = (12288 × 12288) identity kernel ≈
**600 MB**，conv2d 期间不释放。已替换为两次 `Tensor.unfold`（纯 view, 零分配）。

数值等价性由 [`tests/neural/test_patcher_unfold_equivalence.py`](projects/torch_vct/tests/neural/test_patcher_unfold_equivalence.py)
守护：CPU 上 forward bit-exact，CUDA 上 TF32 关闭后 forward bit-exact，backward
gradient 差 < 1 fp32 ulp。

## 8. dev box batch capacity（实测，post aa47ca1）

复现脚本：[`scripts/benchmark_batch_capacity.py`](projects/torch_vct/scripts/benchmark_batch_capacity.py)，
在 dev box 和 training box 上跑出来对比即可。

dev box (RTX 5090 Laptop, 24 GB, bf16 AMP, Vimeo 7-frame septuplet):

| config | B | alloc_GB | sec/step | samples/sec |
|---|---|---|---|---|
| V1 ctx=2 | 1 | 10.17 | 0.340 | 2.95 |
| V1 ctx=2 | **2** | **18.01** | **0.573** | **3.49** ← 实际上限 |
| V1 ctx=2 | 4 | OOM | — | — |
| V2 ctx=2 | 1 | 10.53 | 0.335 | 2.99 |
| V2 ctx=2 | 2 | 18.70 | 0.593 | 3.37 |
| V2 ctx=2 | 4 | OOM | — | — |

V2 只比 V1 多 ~0.7 GB（RoPE buffer + SwiGLU FFN 的开销很小）。**dev box 单 micro-step 0.57s
@ B=2**，effective batch=8 via grad_accum=4 → 2.3s / effective step → 1M effective steps ≈ 26 天。

training box (RTX 6000 Ada, 48 GB) 上的容量预计远高（至少 B=6–8），等那台空了跑一遍 benchmark 就有数。

## 9. 待决策：Phase 0 V1 复现的范围

dev box 上跑完整 1M-step V1 复现需要 26 天 —— 不合理。三个候选方案：

| 选项 | 内容 | 时长 (dev box) | 优势 | 风险 |
|---|---|---|---|---|
| **A** | **跳过 Phase 0 完整复现**，直接 Phase 1 (V2 ctx=2)，V1 baseline 用 paper 公布的 BD-rate 数字 | 0 (省略) | 最高效；马上能拿 V2 数据 | 论文需说明对比方法 |
| **B** | Phase 0 短训练 (~200K 步, ~1.3 天)，拿 V1 早期收敛曲线作内部对比 | ~1.3 天 | 内部一致；不依赖外部数字 | 不是 paper 终值，只能做相对比较 |
| **C** | Phase 0 完整复现 1M 步 | ~26 天 | 最严谨 | dev box 串行 26 天，6000 Ada 空了也只能干等 |

agent 推荐 **选项 A**，理由：

1. VCT paper V1 的 BD-rate 在 UVG / MCL-JCV 上已是公开 baseline。审稿人不会
   要求自己复现，写明"V1 数字取自 [VCT, NeurIPS'22]"即可。
2. dev box 26 天浪费在复现一个已有结果，机会成本是 4–5 个 V2 sub-experiment。
3. 选项 B 的"短训练曲线对比"本质是 noise—— V1 / V2 短训练阶段都没收敛，曲线
   形状的差异可能更多反映初始化随机性而非架构差异。

**决策悬而未定**。这台机器在等 user 拍板。

---

## 6. 数据可用后的下一步

1. 确认 Vimeo 路径结构（`sep_trainlist.txt` / `sequences/<group>/<seq>/im{1..7}.png`）
2. 改 [vimeo.yaml](projects/torch_vct/config/datamodule/vimeo.yaml) 的 `data_dir`
3. 本机跑 Phase 0 `fast_dev_run`（1 step 端到端，B=1，只验 wiring）
4. 跑通后 commit
5. 6000 Ada env 建起来 → 同步数据 + 代码 → 开始 Phase 0 真长跑
