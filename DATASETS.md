# Datasets

VCT V2 项目的数据集本地目录与可用性记录。所有路径都是绝对路径，与各 phase 对应见 [`README.md`](README.md) 和 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)。

## Quick reference

**项目范围（2026-05 修订）**：终点 `context_len=32`，**不再追 64/128**。Kinetics 不需要
全集，5% 随机子集（~12 K videos / ~50–80 GB）就够 Phase 3 训练。所需磁盘总和约 **310 GB**。

| Dataset | Role | Path | Status | Phase |
|---|---|---|---|---|
| Vimeo-90K Septuplet | train | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet` | ✅ 可直接训练 | 0 – 2 |
| Kinetics-400 (5% subset) | train | `/media/zzy/data/kinetics-dataset/k400` | ⚠️ 需先确认目录布局，并抽子集 | 3a, 3b |
| UVG (1080p raw) | eval | `/media/zzy/mydata/UVG` | ⏳ YUV，需抽帧成 PNG | eval all |
| UVG (720p raw) | eval | `/media/zzy/mydata/UVG_720p` | ⏳ YUV，需抽帧 | optional |
| ~~MCL-JCV~~ | ~~eval~~ | `/media/zzy/mydata/MCL-JCV` | ❌ datamodule 缺失，降为 out of scope | — |

---

## Vimeo-90K Septuplet（训练主力 — Phase 0..2）

- **路径**：`/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet`
- **结构**：
  - `sep_trainlist.txt` — 64612 行
  - `sep_testlist.txt`  — 7824 行
  - `sequences/<group>/<seq>/im1.png` … `im7.png`
- **总大小**：约 231 G
- **每条样本 shape**：`(7, 3, 256, 256)`（datamodule 默认 256×256 random crop）
- **datamodule**：`projects/torch_vct/datamodules/vimeo.py` → 调用
  `neuralcompression.data.Vimeo90kSeptuplet(as_video=True, frames_per_group=7)`，
  config 在 `projects/torch_vct/config/datamodule/vimeo.yaml`
- **context_len 上限**：返回 7 帧 ⇒ 最多用 6 帧 prev + 1 current ⇒ `context_len ≤ 6`。
  更长上下文必须切到 Kinetics。
- **CLI 用法**：
  ```bash
  datamodule=vimeo \
    "datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'"
  ```
  路径里的 `()` 会被 hydra override grammar 当语法符号，必须 **shell 外层双引号 + hydra 内层单引号**。

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
| Vimeo Triplet | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_triplet` | 当前 datamodule 只读 septuplet |
| ImageNet | `/media/zzy/mydata/imagenet1k` | 仅 13 M，明显不完整 |
| Kodak | `/media/zzy/mydata/DDCM/test_images` | 只有 kodim01/04/23 等少量，不完整 24 张 |

---

## Shell quoting cheat sheet

Hydra 把括号 `(` `)` 当 override grammar 的语法符号，shell 单引号会被剥掉，
所以 `'datamodule.data_dir=/path/with(parens)'` 在 shell 后只剩裸字符串，hydra 会报错。

✅ **正确写法**（外层 shell 双引号 + 内层 hydra 单引号）：
```bash
"datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'"
```

❌ **错误**：
```bash
'datamodule.data_dir=/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'
datamodule.data_dir=/media/zzy/mydata/vimeo-90K\(3F-7F\)/vimeo_septuplet
```

或者一劳永逸：symlink 到无括号路径。
