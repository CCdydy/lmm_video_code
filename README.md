# Level C: LLM-style ACA-RT

最终方案定为 **Level C：LLM-style ACA-RT**。

> 在 frozen DCVC-RT 上，用 LLM-style 长上下文机制替换/增强原来的 temporal entropy
> prior，让每个 latent token 自适应选择看最近 2 帧、8 帧还是 32 帧的历史 context，
> 从而降低 content latent 的熵编码码率。

- Level C 方案见 [`ACA_RT_LEVEL_C.md`](ACA_RT_LEVEL_C.md)
- 当前机器 / env / 数据路径见 [`PLATFORM.md`](PLATFORM.md)
- 数据集路径与可用性见 [`DATASETS.md`](DATASETS.md)
- VCT 长上下文方案背景见 [`ARCHITECTURE.md`](ARCHITECTURE.md) 和
  [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)

---

## Current Direction

```text
DCVC-RT backbone 冻结
    ↓
保留 y / z latent、quantization、decoder、bitstream 主结构
    ↓
新增 DPB memory：最近 32 帧 decoded latent/context features
    ↓
历史 features → K/V projections + RoPE + KV cache
当前 frame query feature → Q projection
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

核心贡献：

| ID | Contribution |
|---|---|
| C1 | Adaptive Context Attention：每个 latent token 自适应选择 temporal context length |
| C2 | Multi-scope Temporal Memory Prior：last-2 / last-8 / last-32 的长程 memory |
| C3 | KV-Cached Efficient Context Modeling：历史帧 K/V 只计算一次并缓存 |

训练策略：不从头训练 DCVC-RT，不重训 encoder/decoder。先用 pretrained DCVC-RT dump
`y_hat`、`z_hat`、original context、previous decoded features，只训练新增 ACA entropy
prior：

```text
loss = NLL(y_hat) + scope cost regularization
```

实验顺序：

1. DCVC-RT baseline
2. fixed last-2 / last-8 / last-32 context
3. multi-scope attention without gate
4. ACA per-token gate
5. scope cost regularization
6. KV cache speed test

当前状态：

```text
Step 0a: passed within locally verifiable scope
    DCVC-RT official inference on UVG_1080p completed:
    7 sequences × 4 rate points = 28 jobs
    28 .bin + 28 .json under /home/zzy/Desktop/DCVC/out_bin/UVG
    RTX 6000 Ada speed: ~118.1 fps encode / ~101.5 fps decode

Step 0b: not passed yet
    within-sequence signal exists
    attention is unstable across seeds
    cross-sequence generalization fails

Next:
    run multi-sequence Step 0b+ before locking architecture or entering Step 1
```

因此现在不直接进 Step 1，也不切 Pivot 3。下一轮 probe 必须覆盖多条 sequence，并同时比较
identity、mean-pool K=8、naive attention K=8、stabilized attention K=8。

---

## VCT V2 Background Status

**项目实际目标范围（2026-05 修订）**：终点是 **context_len ≤ 32**，**不再追 64 / 128**。
原 Phase 4（hybrid 激活）和 Phase 5（KV-cache + Mamba）降级为 out of scope，
对应代码（`modern_blocks.MambaBlock`、`LongCtxJointEncoder` 的 hybrid 分支）保留但训练
路径不会触发 —— ctx ≤ 32 时序列 ≤ 2048 tokens，FlashAttention-2 完整二次注意力足够。

| Phase | 配置 | 状态 |
|---|---|---|
| 0  | V1, `context_len=2` | `fast_dev_run` 通过 @ current 6000 Ada |
| 1a | V2 enc+dec, `context_len=2`, single-GPU **non-DDP** | `fast_dev_run` 通过 @ current 6000 Ada |
| 1b | V2 enc+dec, `context_len=2`, DDP | 当前单卡 6000 Ada 无法验证多 GPU NCCL |
| 2  | V2, `context_len=4..6` (Vimeo 上限) | 等 Phase 1 |
| 3a | V2, `context_len=16` (切 Kinetics 5% subset) | 等 |
| 3b | V2, `context_len=32`（**项目终点**） | 等 |
| ~~4~~ | ~~`context_len=64`~~ | ❌ out of scope |
| ~~5~~ | ~~`context_len=128`~~ | ❌ out of scope |

**当前 6000 Ada 实测 batch 上限**（bf16 AMP，`--steps 1` quick sweep）：

- V1 ctx=2：B=5 / 41.58 GB peak / 1.05 sec/step；B=6 OOM
- V2 ctx=2：B=4 / 38.46 GB peak / 0.86 sec/step；B=5 OOM
- ELIC per-frame 激活仍是主要 footprint

完整数据 + 复现脚本见 [ARCHITECTURE.md §8](ARCHITECTURE.md)。

**训练规模预估**：以当前 RTX 6000 Ada 48 GB 为主机，V2 ctx=2 的 quick sweep
吞吐约 4.65 samples/sec @ B=4；长训前应再用更多 steps 复测稳定吞吐。所需数据集磁盘
约 **160 GB+**（Vimeo 全集 + Kinetics-400 5% 子集 + UVG 抽帧），完整 Kinetics 本机已有约
439 GB。详见 [`DATASETS.md`](DATASETS.md)。

**已修复**：V2 路径下 DDP setup 阶段，RoPE 预计算的 `freqs_cis` 原本是 complex64 buffer，
会在 `_sync_module_states` 广播时触发 NCCL `ComplexFloat` 不支持。当前已在
`modern_blocks.py` 中拆成 `(real, imag)` 两个 float32 buffer，`apply_rotary_emb` 内部
用 `torch.complex(...)` 重组。当前单卡 6000 Ada `fast_dev_run` 已验证通过；多 GPU DDP
仍需在多卡环境复验。

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

## Current Platform

当前 checkout 路径：`/home/zzy/Desktop/NeuralCompression`。

| Item | Current value |
|---|---|
| Host | `ZZY-Desktop` |
| GPU | NVIDIA RTX 6000 Ada Generation, sm_89, 48 GB |
| Driver | 570.211.01 |
| Training env | `/home/zzy/anaconda3/envs/torch_vct/bin/python` |
| Base env | `/home/zzy/anaconda3/bin/python` |

`torch_vct` env 当前版本：Python 3.10.20 / torch 2.1.2+cu121 /
pytorch-lightning 2.1.4 / torchmetrics 1.3.0 / hydra-core 1.3.2 /
compressai 1.2.8 / pytorchvideo 0.1.5。这个 env 可用于训练入口，但目前缺
`pytest`；默认 base env 可跑单测，但缺 Lightning/Hydra/PyTorchVideo，不适合训练。

完整平台记录见 [`PLATFORM.md`](PLATFORM.md)。

### Optional env rebuild

如果要重建一个更新的 `torch_vct` env（例如 torch 2.7.1/cu128），步骤如下。当前
RTX 6000 Ada 不强制需要 sm_120 支持；升级前应先保留现有 env 作为可回退基线。

```bash
# 1. 建 env
/home/zzy/anaconda3/bin/conda create -n torch_vct_new python=3.10 pip -c conda-forge -y

# 2. 钉 setuptools < 81（setuptools 81 删了 pkg_resources，会牵连若干旧包）
/home/zzy/anaconda3/envs/torch_vct_new/bin/pip install 'setuptools<81'

# 3. torch 2.7.1 + cu128
/home/zzy/anaconda3/envs/torch_vct_new/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1

# 4. 科学栈（hydra/lightning/compressai/pytorchvideo/…）
/home/zzy/anaconda3/envs/torch_vct_new/bin/pip install \
  'numpy<2' 'scipy<=1.11.1' \
  'pytorch-lightning>=2.4,<2.6' 'torchmetrics>=1.5,<2' \
  hydra-core==1.3.2 'omegaconf>=2.3,<3' \
  compressai==1.2.8 pytorchvideo==0.1.5 \
  fvcore lpips DISTS-pytorch torch-fidelity \
  tqdm pillow wandb av pytest pytest-timeout

# 5. 项目代码（editable）。如果 .git 没 tag 或仓库刚 init，
#    setuptools_scm 拿不到版本号，必须用 SETUPTOOLS_SCM_PRETEND_VERSION 兜底：
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NEURALCOMPRESSION=0.3.0 \
  /home/zzy/anaconda3/envs/torch_vct_new/bin/pip install -e /home/zzy/Desktop/NeuralCompression
```

**已知兼容性 shim**：pytorchvideo 0.1.5（2022 年后未更新）在 `transforms/augmentations.py`
里 `import torchvision.transforms.functional_tensor as F_t`，但 torchvision ≥ 0.17
已把该模块改名为 `_functional_tensor`（私有路径）。如果升级 torchvision，需要在
新 env 里放 `sitecustomize.py`，把 `_functional_tensor` 别名回 `functional_tensor`。**升级 pytorchvideo
或重建 env 后需要重写这个 shim 文件**，否则 `from pytorchvideo.transforms import ...`
直接 ImportError。

---

## Running

### V1 baseline `fast_dev_run`（验证过通过）

```bash
cd /home/zzy/Desktop/NeuralCompression/projects/torch_vct
/home/zzy/anaconda3/envs/torch_vct/bin/python model_train.py \
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

### Unit tests

当前 base env 已验证可以跑项目单测：

```bash
cd /home/zzy/Desktop/NeuralCompression
python -m pytest projects/torch_vct/tests -q
```

### V2 enc+dec sanity

```bash
cd /home/zzy/Desktop/NeuralCompression/projects/torch_vct
/home/zzy/anaconda3/envs/torch_vct/bin/python model_train.py \
  datamodule=vimeo \
  "datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'" \
  trainer.fast_dev_run=true \
  ngpu=1 \
  num_workers_per_task=0 \
  model.use_v2_encoder=true \
  model.use_v2_decoder=true \
  'hydra.run.dir=/tmp/torch_vct_fastdev_v2_current'
```

---

## Code Modifications Applied

| 文件 | 改动 | 原因 |
|---|---|---|
| `projects/torch_vct/neural/modern_blocks.py` | 新增：RMSNorm / SwiGLU_FFN / RoPE / FlashAttnBlock / LongCtxJointEncoder / MambaBlock；RoPE buffer 拆成 real/imag float buffer | V2 基础设施；避免 NCCL 广播 complex buffer |
| `.../neural/entropy_model_layers.py` | 末尾追加 `TransformerBlockV2`, `TransformerV2`（原类全部保留） | V2 transformer stack |
| `.../neural/entropy_model.py` | 加 `use_v2_encoder` / `use_v2_decoder` 开关；`_get_encoded_seqs` 支持任意 `context_len`，开头帧不足时 left-pad 最早 latent | V2 接线 + 多帧支持 |
| `.../model_pipeline.py` | 把 v2 flags 透传给 `VCTEntropyModel` | 配置传递 |
| `.../model_lightning.py:47` | 删掉 `training_step` 的 `optimizer_idx` 参数 | PL 1.7 → 2.x：有此参数会被推断为 multi-optim |
| `.../config/train_config.yaml` | 显式默认 `context_len=2 / use_v2_encoder=False / use_v2_decoder=False` | baseline 行为清晰 |

V1 路径与原 VCT bit-exact，所有 V2 改动都在 `if use_v2_*:` 分支内。

---

## License

MIT，见 [`LICENSE`](LICENSE)。
