# Datasets

VCT V2 项目的数据集本地目录与可用性记录。所有路径都是绝对路径，与各 phase 对应见 [`README.md`](README.md) 和 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)。

## Quick reference

**项目范围（2026-05 修订）**：终点 `context_len=32`，**不再追 64/128**。Kinetics 不需要
全集，5% 随机子集（~12 K videos / ~50–80 GB）就够 Phase 3 训练。所需磁盘总和约 **160 GB**
（Vimeo 82 GB + Kinetics 5% 子集 50–80 GB + UVG 抽帧 ~30 GB）。

| Dataset | Role | Path（当前 6000 Ada 平台） | Status | Phase |
|---|---|---|---|---|
| Vimeo-90K Septuplet | train | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet` | 已解压 | 0 – 2 |
| Kinetics-400 | train | `/media/zzy/data/kinetics-dataset/k400` | 已就绪，约 439 GB；Phase 3 仍建议抽 5% 子集 | 3a, 3b |
| UVG (1080p raw) | eval | `/media/zzy/mydata/UVG` | YUV 已就绪，需抽帧成 PNG | eval all |
| UVG (720p raw) | eval | `/media/zzy/mydata/UVG_720p` | YUV 已就绪，需抽帧 | optional |
| DCVC-RT UVG baseline outputs | baseline | `/home/zzy/Desktop/DCVC/out_bin/UVG` | 28 `.bin` + 28 `.json` 已观察到 | ACA Exp 0a |
| ~~MCL-JCV~~ | ~~eval~~ | — | ❌ datamodule 缺失，out of scope | — |

**冷存档 / 原始包**：Vimeo zip 在 `/media/zzy/mydata/vimeo-90K(3F-7F)/`，
Kinetics tarball 备份在 `/media/zzy/data/kinetics-dataset/k400_targz`。

---

## Vimeo-90K Septuplet（训练主力 — Phase 0..2）

- **当前路径**：`/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet`
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
  datamodule=vimeo "datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'"
  ```
  路径里有括号，必须使用外层 shell 双引号 + 内层 hydra 单引号。

## Kinetics-400 子集（长上下文 — Phase 3a / 3b）

- **路径**：`/media/zzy/data/kinetics-dataset/k400`
- **完整规模**：train 241258 / val 19881 / test 38685 个 mp4，约 439 G
- **本项目只需 5% 子集**（约 12 K videos / **50–80 GB**）。每个视频 ≥ 3 秒即可（30 fps × 3 = 90 帧，能切出 ≥ 2 个 ctx=32 训练窗口）。整集对 ctx ≤ 32 的训练规模冗余太多。
- **另有 tarball 备份**：`/media/zzy/data/kinetics-dataset/k400_targz`（约 436 G）
- **datamodule**：`projects/torch_vct/datamodules/kinetics.py` → 调用
  `pytorchvideo.data.Kinetics(data_path=data_dir/train)`，**pytorchvideo 默认期望
  `train/<label>/*.mp4` 的二级目录结构**。
- **使用前必做**：
  1. `k400/train` 当前存在大量 mp4；正式跑 Phase 3 前仍需确认是否满足
     `pytorchvideo.data.Kinetics(data_path=data_dir/train)` 的标签目录假设。
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

## DCVC-RT UVG baseline outputs（ACA Exp 0a）

- **路径**：`/home/zzy/Desktop/DCVC/out_bin/UVG`
- **状态**：已观察到 28 个 `.bin` 和 28 个 `.json`，对应 7 个 UVG 1080p sequence ×
  4 个 rate points (`q0`, `q21`, `q42`, `q63`)。
- **用途**：这是 ACA-RT 的 Step 0a 官方 baseline，不是 ACA 设计验证。后续 Level-2
  bitstream evaluation 要用这些 `.bin` 作为同 `y_hat` 条件下的码率参照。
- **注意**：这些输出属于外部 DCVC checkout 的运行产物，不纳入本仓库。

## ~~MCL-JCV（评测）~~ — 项目范围外

- **路径**：`/media/zzy/mydata/MCL-JCV/`，约 83 G，3121 个视频文件
- **状态**：datamodule 不存在，且本项目终点 ctx=32 用 UVG 一个评测集已能给出 BD-rate 曲线，不再投入精力写 MCL-JCV datamodule。
- 如未来要扩展，参考做法仍是：ffmpeg 抽帧 → 沿用 `ImageFolder` 模式 → 复用 [datamodules/uvg.py](projects/torch_vct/datamodules/uvg.py) 模板。

---

## 已排除（不要用）

| Dataset | 路径 | 原因 |
|---|---|---|
| Vimeo Triplet | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_triplet*` | 当前 datamodule 只读 septuplet，无需解压 |
| ImageNet | (不在本机) | 之前文档提及过的 13 MB 残缺集，不可用 |
| Kodak | (不在本机) | 之前文档提及过的部分图，不完整 |

---

## Shell quoting cheat sheet

当前 Vimeo 路径含有括号 `(` `)`，Hydra override 必须嵌套引号：

✅ **正确**（外层 shell 双引号 + 内层 hydra 单引号）：
```bash
"datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'"
```

❌ **错误**（shell 单引号被剥掉，hydra 看到裸括号）：
```bash
'datamodule.data_dir=/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'
datamodule.data_dir=/media/zzy/mydata/vimeo-90K\(3F-7F\)/vimeo_septuplet
```

或一劳永逸：symlink 到无括号路径。
