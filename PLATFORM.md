# Current Platform

This file records the platform currently used for this checkout:
`/home/zzy/Desktop/NeuralCompression`.

## Hardware

| Item | Value |
|---|---|
| Host | `ZZY-Desktop` |
| OS | Ubuntu 22.04, Linux 6.8 |
| GPU | NVIDIA RTX 6000 Ada Generation |
| CUDA capability | `sm_89` |
| VRAM | 49140 MiB |
| Driver | 570.211.01 |

This is now the primary training/debug platform. Older docs may mention a
separate RTX 5090 laptop dev box; treat those measurements as historical
reference only.

## Repository

| Item | Value |
|---|---|
| Checkout | `/home/zzy/Desktop/NeuralCompression` |
| Active plan | Level C: LLM-style ACA-RT |
| Plan doc | `ACA_RT_LEVEL_C.md` |
| Reusable code project | `projects/torch_vct` |
| Main training entry | `projects/torch_vct/model_train.py` |
| Default config | `projects/torch_vct/config/train_config.yaml` |

## Python Environments

### Training Env

Use this environment for VCT training and Lightning/Hydra runs:

```bash
/home/zzy/anaconda3/envs/torch_vct/bin/python
```

Observed versions:

| Package | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.1.2+cu121 |
| torchvision | 0.16.2+cu121 |
| pytorch-lightning | 2.1.4 |
| torchmetrics | 1.3.0 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| compressai | 1.2.8 |
| pytorchvideo | 0.1.5 |

Known gap: `pytest` is not installed in this environment yet.

### Base Env

The default shell currently resolves `python` to:

```bash
/home/zzy/anaconda3/bin/python
```

Observed versions: Python 3.12.7, torch 2.5.1+cu121. This environment can run
the current unit tests, but it is missing Lightning, Hydra, torchmetrics, and
pytorchvideo, so it is not a training environment.

## Data Paths

| Dataset | Path | Status |
|---|---|---|
| Vimeo-90K Septuplet | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet` | Available |
| Vimeo-90K Septuplet zip | `/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet.zip` | Available |
| Kinetics-400 | `/media/zzy/data/kinetics-dataset/k400` | Available, about 439 GB |
| Kinetics-400 tarballs | `/media/zzy/data/kinetics-dataset/k400_targz` | Available, about 436 GB |
| UVG 1080p raw | `/media/zzy/mydata/UVG` | Available, about 12 GB |
| UVG 720p raw | `/media/zzy/mydata/UVG_720p` | Available, about 914 MB |
| DCVC UVG outputs | `/home/zzy/Desktop/DCVC/out_bin/UVG` | Available, reference outputs only |

Vimeo's parent path contains parentheses. When passing it through Hydra, use a
quoted override:

```bash
"datamodule.data_dir='/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet'"
```

## Quick Commands

Unit tests with the currently working base env:

```bash
cd /home/zzy/Desktop/NeuralCompression
python -m pytest projects/torch_vct/tests -q
```

V1 fast dev run with the training env:

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

V2 encoder+decoder fast dev run:

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

## Verified Now

| Check | Result |
|---|---|
| V1 `fast_dev_run` on Vimeo | Passed |
| V2 enc+dec `fast_dev_run` on Vimeo | Passed |
| `python -m pytest projects/torch_vct/tests -q` | 32 passed |
| Batch quick sweep | V1 B=5 passes/B=6 OOM; V2 B=4 passes/B=5 OOM |

The test command currently uses the base env because `pytest` is not installed
in `/home/zzy/anaconda3/envs/torch_vct`.

Current RTX 6000 Ada quick benchmark (`bf16`, `--steps 1`):

| Config | B | alloc_GB | reserved_GB | sec/step | samples/sec |
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
