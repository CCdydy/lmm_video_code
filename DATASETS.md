# Datasets

VCT V2 项目的数据集本地目录与可用性记录。所有路径都是绝对路径，与各 phase 对应见 [`README.md`](README.md) 和 [`VCT_V2_Design_Log.md`](VCT_V2_Design_Log.md)。

## Quick reference

| Dataset | Role | Path | Status | Phase |
|---|---|---|---|---|
| Vimeo-90K Septuplet | train | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet` | ✅ 可直接训练 | 0 – 2 |
| Kinetics-400 | train | `/media/zzy/data/kinetics-dataset/k400` | ⚠️ 需先确认目录布局 | 3 – 5 |
| UVG (1080p raw) | eval | `/media/zzy/mydata/UVG` | ⏳ YUV，需抽帧成 PNG | eval all |
| UVG (720p raw) | eval | `/media/zzy/mydata/UVG_720p` | ⏳ YUV，需抽帧 | optional |
| MCL-JCV | eval | `/media/zzy/mydata/MCL-JCV` | ⏳ 没 datamodule | eval all |

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

## Kinetics-400（长上下文 — Phase 3..5）

- **路径**：`/media/zzy/data/kinetics-dataset/k400`
- **规模**：train 241258 / val 19881 / test 38685 个 mp4，约 439 G
- **另有 tarball 备份**：`/media/zzy/data/kinetics-dataset/k400_targz`（约 436 G）
- **datamodule**：`projects/torch_vct/datamodules/kinetics.py` → 调用
  `pytorchvideo.data.Kinetics(data_path=data_dir/train)`，**pytorchvideo 默认期望
  `train/<label>/*.mp4` 的二级目录结构**。
- **🔴 使用前必做**：确认 `k400/train/` 下是直接装 mp4，还是按 label 分文件夹。
  如果是平铺的 mp4，需要做下面之一：
  1. 改写 datamodule 用 `LabeledVideoDataset` + 自定义路径列表
  2. 在 `train/` 下加一层伪 label 目录（symlink 或重组）

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

## MCL-JCV（评测）

- **路径**：`/media/zzy/mydata/MCL-JCV/`，约 83 G，3121 个视频文件
- **datamodule**：**不存在** —— 需要写一个。结构类似 UVG，建议沿用 `ImageFolder` 模式：
  1. 用 ffmpeg 把每个视频抽帧成 PNG 子目录
  2. 在 `projects/torch_vct/datamodules/mcl_jcv.py` 复用 UVG datamodule 模板
  3. 加 `projects/torch_vct/config/test_datamodule/mcl_jcv.yaml`

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
