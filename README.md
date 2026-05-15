# VCT V2: Fill the GOP

把 VCT (Mentzer et al., NeurIPS 2022) 的 joint spatiotemporal self-attention
从 **2 帧** 扩展到 **整个 GOP（32 帧）**，用 FlashAttention-2 + RoPE + RMSNorm + SwiGLU
让"多帧联合自注意力"这个早期被算力限制锁死的设计可以真正放大。

Main contribution metric：**GOP utilization 6% (2/32) → 100% (32/32)**。

- 架构一页 reference 见 [`ARCHITECTURE.md`](ARCHITECTURE.md)
- 详细设计 / novelty / related work 见 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)
- 数据集路径与可用性见 [`DATASETS.md`](DATASETS.md)

---

## Status

**项目实际目标范围（2026-05 修订）**：终点是 **context_len ≤ 32**，**不再追 64 / 128**。
原 Phase 4（hybrid 激活）和 Phase 5（KV-cache + Mamba）降级为 out of scope，
对应代码（`modern_blocks.MambaBlock`、`LongCtxJointEncoder` 的 hybrid 分支）保留但训练
路径不会触发 —— ctx ≤ 32 时序列 ≤ 2048 tokens，FlashAttention-2 完整二次注意力足够。

| Phase | 配置 | 状态 |
|---|---|---|
| 0  | V1, `context_len=2` | ✅ `fast_dev_run` 通过 @ dev box (B=1, Vimeo) |
| 1a | V2 enc+dec, `context_len=2`, single-GPU **non-DDP** | ⚠️ 待测 |
| 1b | V2 enc+dec, `context_len=2`, DDP | 🟡 NCCL/complex 需 6000 Ada 多 GPU 上确认 |
| 2  | V2, `context_len=4..6` (Vimeo 上限) | 等 Phase 1 |
| 3a | V2, `context_len=16` (切 Kinetics 5% subset) | 等 |
| 3b | V2, `context_len=32`（**项目终点**） | 等 |
| ~~4~~ | ~~`context_len=64`~~ | ❌ out of scope |
| ~~5~~ | ~~`context_len=128`~~ | ❌ out of scope |

**dev box 实测 batch 上限**（post commit aa47ca1，bf16 AMP）：

- V1 ctx=2：B=2 / 18.0 GB peak / 0.57 sec/step
- V2 ctx=2：B=2 / 18.7 GB peak / 0.59 sec/step
- B=4 OOM；ELIC per-frame 激活是主要 footprint

完整数据 + 复现脚本见 [ARCHITECTURE.md §8](ARCHITECTURE.md)，跑一遍 6000 Ada 即可对比。
**Phase 0 V1 完整复现耗时 ~26 天 @ dev box，已挂起，等 user 决策选项 A/B/C
（见 [ARCHITECTURE.md §9](ARCHITECTURE.md)）**。

**训练规模预估**：单卡 RTX 5090 Laptop (~110 TFLOPS bf16)，从 Phase 0 走到 Phase 3b
约需 **2–3 周纯训练时间**。所需数据集磁盘约 **310 GB**（Vimeo 全集 + Kinetics-400
随机 5% 子集 + UVG 抽帧）。详见 [`DATASETS.md`](DATASETS.md)。

**已知开放问题**：V2 路径下 DDP setup 阶段，RoPE 预计算的 `freqs_cis` 是 complex64 buffer，
被 `_sync_module_states` 广播时 NCCL 报错。修法：在 `modern_blocks.py` 里把它拆成
`(real, imag)` 两个 float32 buffer 存，`apply_rotary_emb` 内部 `torch.complex(...)` 重组。
（torch 已升到 2.7.1，要先验证此 bug 是否还存在再决定是否修。）

---

## Repo Layout

```
NeuralCompression/
├── projects/torch_vct/           ← 唯一活跃项目
│   ├── neural/                   ← V1 + V2 transformer / RoPE / Mamba / 量化层
│   │   ├── entropy_model.py
│   │   ├── entropy_model_layers.py
│   │   ├── modern_blocks.py      ← V2 新增（RoPE / SwiGLU / FlashAttn / Mamba）
│   │   ├── bottlenecks.py
│   │   ├── transforms.py         ← ELIC analysis / synthesis
│   │   └── patcher.py
│   ├── datamodules/              ← Vimeo / Kinetics / UVG
│   ├── config/                   ← hydra YAML
│   ├── utils/  tests/
│   ├── model_pipeline.py         ← VCTPipeline
│   ├── model_lightning.py        ← LightningModule（已 PL-2.x 适配）
│   └── model_train.py            ← 入口
├── neuralcompression/            ← 上游核心包；本项目只用 Vimeo90kSeptuplet
├── VCT_V2_Design_Log.md
├── DATASETS.md
└── setup.{py,cfg} / pyproject.toml
```

---

## Environment

**双机分工**（详见 [ARCHITECTURE.md §4](ARCHITECTURE.md)）：

| | dev box (本机) | training box |
|---|---|---|
| GPU | RTX 5090 Laptop, sm_120, 24 GB | RTX 6000 Ada, sm_89, 48 GB |
| 用途 | sanity / wiring / debug | 全部真实训练长跑 |
| env 状态 | ✅ 已建好 | ⏳ 待 git pull 后按下方步骤重建 |

### dev box: `/home/zzy/miniforge3/envs/torch_vct/`

- 关键版本：Python 3.10 / **torch 2.7.1+cu128** / **pytorch-lightning 2.5.6** /
  **torchmetrics 1.9.0** / hydra-core 1.3.2 / numpy 1.26.4 / scipy 1.11.1 /
  setuptools 80.10.2 / compressai 1.2.8 / pytorchvideo 0.1.5
- torch 2.7.1 的 cu128 wheel 含原生 sm_120 binary，**不再需要 PTX-JIT**（架构列表里 `sm_120`
  + `compute_120` 都在）。
- 旧 docs 里钉的 torch 2.1.2 / lightning 2.1.4 不适用 sm_120：torch ≤ 2.6 没有 sm_120 binary 也无法 JIT。

### training box: 同样照下方步骤建 env

sm_89 (Ada) 是 mature 架构，cu128 wheel 完全兼容；上面的版本组合直接复用。**额外**
可以再装 `flash_attn` PyPI 包（sm_89 有预编译 wheel），dev box 上 sm_120 的 wheel
还不稳定所以这台不装。`flash_attn` 包只对 Phase 3c 的 ALiBi in-kernel bias 有用，
Phase 0–3b 主路径用 PyTorch SDPA flash backend 就够。

**重建 env 的步骤**（顺序敏感）：

```bash
# 1. 建 env
/home/zzy/miniforge3/bin/mamba create -n torch_vct python=3.10 pip -c conda-forge -y

# 2. 钉 setuptools < 81（setuptools 81 删了 pkg_resources，会牵连若干旧包）
/home/zzy/miniforge3/envs/torch_vct/bin/pip install 'setuptools<81'

# 3. torch 2.7.1 + cu128（必须从 pytorch 官方 index 拉 cu128 轮子）
/home/zzy/miniforge3/envs/torch_vct/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1

# 4. 科学栈（hydra/lightning/compressai/pytorchvideo/…）
/home/zzy/miniforge3/envs/torch_vct/bin/pip install \
  'numpy<2' 'scipy<=1.11.1' \
  'pytorch-lightning>=2.4,<2.6' 'torchmetrics>=1.5,<2' \
  hydra-core==1.3.2 'omegaconf>=2.3,<3' \
  compressai==1.2.8 pytorchvideo==0.1.5 \
  fvcore lpips DISTS-pytorch torch-fidelity \
  tqdm pillow wandb av pytest pytest-timeout

# 5. 项目代码（editable）。如果 .git 没 tag 或仓库刚 init，
#    setuptools_scm 拿不到版本号，必须用 SETUPTOOLS_SCM_PRETEND_VERSION 兜底：
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NEURALCOMPRESSION=0.3.0 \
  /home/zzy/miniforge3/envs/torch_vct/bin/pip install -e /home/zzy/NeuralCompression
```

**已知兼容性 shim**：pytorchvideo 0.1.5（2022 年后未更新）在 `transforms/augmentations.py`
里 `import torchvision.transforms.functional_tensor as F_t`，但 torchvision ≥ 0.17
已把该模块改名为 `_functional_tensor`（私有路径）。env 内已放了
[`sitecustomize.py`](/home/zzy/miniforge3/envs/torch_vct/lib/python3.10/site-packages/sitecustomize.py)
在 Python 启动时把 `_functional_tensor` 别名回 `functional_tensor`。**升级 pytorchvideo
或重建 env 后需要重写这个 shim 文件**，否则 `from pytorchvideo.transforms import ...`
直接 ImportError。

---

## Running

### V1 baseline `fast_dev_run`（验证过通过）

```bash
cd /home/zzy/NeuralCompression/projects/torch_vct
/home/zzy/miniforge3/envs/torch_vct/bin/python model_train.py \
  datamodule=vimeo \
  "datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'" \
  trainer.fast_dev_run=true \
  ngpu=1 \
  num_workers_per_task=0 \
  'hydra.run.dir=/tmp/torch_vct_fastdev'
```

**注意 Vimeo 路径里的 `()`**：shell 单引号会被剥掉，hydra 看到裸括号会报
override grammar 错。正确做法是 *shell 外层双引号 + hydra 内层单引号* 嵌套
（如上 `datamodule.data_dir='...'` 整段被外层双引号包住）。

### V2 enc+dec sanity（DDP 修复后跑）

```bash
... model.use_v2_encoder=true model.use_v2_decoder=true ...
```

---

## Code Modifications Applied

| 文件 | 改动 | 原因 |
|---|---|---|
| `projects/torch_vct/neural/modern_blocks.py` | 新增（~393 行）：RMSNorm / SwiGLU_FFN / RoPE / FlashAttnBlock / LongCtxJointEncoder / MambaBlock | V2 基础设施 |
| `.../neural/entropy_model_layers.py` | 末尾追加 `TransformerBlockV2`, `TransformerV2`（原类全部保留） | V2 transformer stack |
| `.../neural/entropy_model.py` | 加 `use_v2_encoder` / `use_v2_decoder` 开关；`_get_encoded_seqs` 支持任意 `context_len`，开头帧不足时 left-pad 最早 latent | V2 接线 + 多帧支持 |
| `.../model_pipeline.py` | 把 v2 flags 透传给 `VCTEntropyModel` | 配置传递 |
| `.../model_lightning.py:47` | 删掉 `training_step` 的 `optimizer_idx` 参数 | PL 1.7 → 2.x：有此参数会被推断为 multi-optim |
| `.../config/train_config.yaml` | 显式默认 `context_len=2 / use_v2_encoder=False / use_v2_decoder=False` | baseline 行为清晰 |

V1 路径与原 VCT bit-exact，所有 V2 改动都在 `if use_v2_*:` 分支内。

---

## License

MIT，见 [`LICENSE`](LICENSE)。
