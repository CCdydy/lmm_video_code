# Datasets

VCT V2 项目的数据集本地目录与可用性记录。所有路径都是绝对路径，与各 phase 对应见 [`README.md`](README.md) 和 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)。

## Quick reference

**项目范围（2026-05 修订）**：终点 `context_len=32`，**不再追 64/128**。Kinetics 不需要
全集，5% 随机子集（~12 K videos / ~50–80 GB）就够 Phase 3 训练。所需磁盘总和约 **160 GB**
（Vimeo 82 GB + Kinetics 5% 子集 50–80 GB + UVG 抽帧 ~30 GB）。

| Dataset | Role | Path（dev box, 本机） | Status | Phase |
|---|---|---|---|---|
| Vimeo-90K Septuplet | train | `/home/zzy/data/vimeo_septuplet/` | ✅ 已解压 | 0 – 2 |
| Kinetics-400 (5% subset) | train | TBD | ⚠️ 需先确认目录布局，并抽子集 | 3a, 3b |
| UVG (1080p raw) | eval | TBD | ⏳ YUV，需抽帧成 PNG | eval all |
| UVG (720p raw) | eval | TBD | ⏳ YUV，需抽帧 | optional |
| ~~MCL-JCV~~ | ~~eval~~ | — | ❌ datamodule 缺失，out of scope | — |

**冷存档**：原始 zip 包在 `/media/zzy/861b729e-7b67-554f-bf33-48fa81660b97/` 这块 SSD 上
（`vimeo_septuplet.zip` 82 GB / `vimeo_triplet.zip` 33 GB —— triplet 我们不用）。

**training box (RTX 6000 Ada)** 上的路径以那台机器为准，本表只记 dev box 状态。

---

## Vimeo-90K Septuplet（训练主力 — Phase 0..2）

- **dev box 路径**：`/home/zzy/data/vimeo_septuplet/`（解压自 SSD 上的 zip）
- **结构**：
  - `sep_trainlist.txt` — 64612 行
  - `sep_testlist.txt`  — 7824 行
  - `readme.txt`
  - `sequences/<group>/<seq>/im{1..7}.png`
- **总大小**：**约 82 GB**（实测，PNG 已是压缩格式，zip 几乎没再压一遍 ——
  之前文档误估为 "231 GB"，是错的）
- **文件数**：733,709（约 64,612 段 × 7 PNG + list 文件 + 目录条目）
- **每条样本 shape**：`(7, 3, 256, 256)`（datamodule 默认 256×256 random crop）
- **datamodule**：`projects/torch_vct/datamodules/vimeo.py` → 调用
  `neuralcompression.data.Vimeo90kSeptuplet(as_video=True, frames_per_group=7)`，
  config 在 `projects/torch_vct/config/datamodule/vimeo.yaml`
- **context_len 上限**：返回 7 帧 ⇒ 最多用 6 帧 prev + 1 current ⇒ `context_len ≤ 6`。
  更长上下文必须切到 Kinetics。
- **CLI 用法**：
  ```bash
  datamodule=vimeo datamodule.data_dir=/home/zzy/data/vimeo_septuplet
  ```
  （新路径没有括号，不需要 hydra override 嵌套引号）

## Kinetics-400 子集（长上下文 — Phase 3a / 3b）

- **路径**：`/media/zzy/data/kinetics-dataset/k400`
- **完整规模**：train 241258 / val 19881 / test 38685 个 mp4，约 439 G
- **本项目只需 5% 子集**（约 12 K videos / **50–80 GB**）。每个视频 ≥ 3 秒即可（30 fps × 3 = 90 帧，能切出 ≥ 2 个 ctx=32 训练窗口）。整集对 ctx ≤ 32 的训练规模冗余太多。
- **另有 tarball 备份**：`/media/zzy/data/kinetics-dataset/k400_targz`（约 436 G）
- **datamodule**：`projects/torch_vct/datamodules/kinetics.py` → 调用
  `pytorchvideo.data.Kinetics(data_path=data_dir/train)`，**pytorchvideo 默认期望
  `train/<label>/*.mp4` 的二级目录结构**。
- **🔴 使用前必做**：
  1. 确认 `k400/train/` 下是平铺 mp4 还是按 label 分文件夹。如果是平铺，要么改写 datamodule 用 `LabeledVideoDataset` + 自定义路径列表，要么在 `train/` 下加一层伪 label 目录（symlink 或重组）。
  2. 抽 5% 子集：`find $K400/train -name '*.mp4' | shuf -n 12000 | ...` 后软链到子集目录。

## UVG（评测）

- **1080p raw**：`/media/zzy/mydata/UVG/`，7 个 `.yuv` 原始视频，约 12 G
- **720p raw**：`/media/zzy/mydata/UVG_720p/`，7 个 `.yuv`，约 914 M
- **datamodule**：`projects/torch_vct/datamodules/uvg.py` 用
  `torchvision.datasets.ImageFolder` 读图像，**不读 YUV**。
- **需要先抽帧**：每个视频按 300 帧切（6 个 600 帧的视频拆成两段 + 1 个 300 帧 = 13 个子目录）：
  ```bash
  ffmpeg -f rawvideo -pix_fmt yuv420p -s 1920x1080 -r 120 \
    -i Beauty_1920x1080_120fps_420_8bit_YUV.yuv \
    -vframes 300 frames_beauty_part1/im_%04d.png
  ```
  最终目录结构应为 `<root>/<video_part>/*.png`，13 个 part 子目录。
- **冗余副本**（不要用，疑似旧拷贝）：
  - `/media/zzy/mydata/UVG Dataset(1080p)/`
  - `/media/zzy/data/Ｗan_experiment_data/data/uvg/`

## ~~MCL-JCV（评测）~~ — 项目范围外

- **路径**：`/media/zzy/mydata/MCL-JCV/`，约 83 G，3121 个视频文件
- **状态**：datamodule 不存在，且本项目终点 ctx=32 用 UVG 一个评测集已能给出 BD-rate 曲线，不再投入精力写 MCL-JCV datamodule。
- 如未来要扩展，参考做法仍是：ffmpeg 抽帧 → 沿用 `ImageFolder` 模式 → 复用 [datamodules/uvg.py](projects/torch_vct/datamodules/uvg.py) 模板。

---

## 已排除（不要用）

| Dataset | 路径 | 原因 |
|---|---|---|
| Vimeo Triplet | `/media/zzy/861b729e-.../vimeo_triplet.zip` (SSD 上) | 当前 datamodule 只读 septuplet，无需解压 |
| ImageNet | (不在本机) | 之前文档提及过的 13 MB 残缺集，不可用 |
| Kodak | (不在本机) | 之前文档提及过的部分图，不完整 |

---

## Shell quoting cheat sheet（历史，目前不触发）

dev box 的解压目标 `/home/zzy/data/vimeo_septuplet/` 没有特殊字符，hydra
直接 `datamodule.data_dir=/home/zzy/data/vimeo_septuplet` 即可。

如果未来 path 里出现括号 `(` `)` 等 hydra override grammar 符号（之前老路径
`vimeo-90K(3F-7F)` 就是这情况），写法：

✅ **正确**（外层 shell 双引号 + 内层 hydra 单引号）：
```bash
"datamodule.data_dir='/some/path/with(parens)/data'"
```

❌ **错误**（shell 单引号被剥掉，hydra 看到裸括号）：
```bash
'datamodule.data_dir=/some/path/with(parens)/data'
datamodule.data_dir=/some/path/with\(parens\)/data
```

或一劳永逸：symlink 到无括号路径。
