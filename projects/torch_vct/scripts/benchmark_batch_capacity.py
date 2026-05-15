"""Profile peak GPU memory and per-step time at increasing batch sizes
for both V1 and V2 entropy models at context_len=2.

Designed to be cross-machine: run the same script on any target GPU to record
the real, post-patcher-fix capacity.

Reference numbers measured on the old dev box (5090 LP, bf16-AMP, post-aa47ca1):

    config       B   alloc_GB   sec/step
    V1 ctx=2     1     10.17     0.340
    V1 ctx=2     2     18.01     0.573    ← practical limit (24 GB)
    V1 ctx=2     4      OOM
    V2 ctx=2     1     10.53     0.335
    V2 ctx=2     2     18.70     0.593
    V2 ctx=2     4      OOM

Usage:
    cd projects/torch_vct
    python scripts/benchmark_batch_capacity.py \
        --data-dir '/media/zzy/mydata/vimeo-90K(3F-7F)/vimeo_septuplet' \
        --batches 1 2 4 6 8 \
        --precision bf16
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch

# Allow running from inside projects/torch_vct/ or from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def _autocast_ctx(precision: str):
    if precision == "fp32":
        from contextlib import nullcontext

        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    return torch.amp.autocast("cuda", dtype=dtype)


def run_one(B: int, data_dir: str, use_v2: bool, precision: str, n_steps: int):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg_dir = str(_PROJECT_ROOT / "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(
            config_name="train_config",
            overrides=[
                "datamodule=vimeo",
                f"datamodule.data_dir='{data_dir}'",
                f"training_loop.train_batch_size={B}",
                "training_loop.val_batch_size=1",
                f"model.use_v2_encoder={use_v2}",
                f"model.use_v2_decoder={use_v2}",
                "ngpu=1",
                "num_workers_per_task=0",
            ],
        )

    dm = instantiate(cfg.datamodule, pin_memory=False)
    dm.setup(stage=None)
    model = instantiate(cfg.model).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(dm.train_dataloader())

    # 1 warmup step (cuDNN tuning, allocator priming)
    batch = next(it)
    bv = type(batch)(video_tensor=batch.video_tensor.cuda())
    with _autocast_ctx(precision):
        recon, rate_args = model(bv)
        loss = recon.float().mean()
        for arg in rate_args:
            if isinstance(arg, torch.Tensor):
                loss = loss + arg.float().mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(n_steps):
        batch = next(it)
        bv = type(batch)(video_tensor=batch.video_tensor.cuda())
        with _autocast_ctx(precision):
            recon, rate_args = model(bv)
            loss = recon.float().mean()
            for arg in rate_args:
                if isinstance(arg, torch.Tensor):
                    loss = loss + arg.float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    sec_per_step = (time.time() - t0) / n_steps

    out = {
        "B": B,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "sec_per_step": sec_per_step,
        "samples_per_sec": B / sec_per_step,
    }
    del model, opt, batch, bv, recon, rate_args, loss, dm, it
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-dir", required=True, help="Vimeo90k septuplet root")
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8],
        help="batch sizes to sweep",
    )
    p.add_argument(
        "--precision", choices=["fp32", "bf16", "fp16"], default="bf16",
    )
    p.add_argument("--steps", type=int, default=3, help="timed steps per config")
    p.add_argument(
        "--variants",
        nargs="+",
        choices=["v1", "v2"],
        default=["v1", "v2"],
        help="which model variants to bench",
    )
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"(cap {torch.cuda.get_device_capability(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"Precision: {args.precision}, steps/config: {args.steps}")
    print()
    print(f"{'config':>16s}  {'B':>3s}  {'alloc_GB':>9s}  {'reserved_GB':>12s}  "
          f"{'sec/step':>9s}  {'samples/s':>10s}")
    print("-" * 70)

    for variant in args.variants:
        use_v2 = variant == "v2"
        tag = f"{variant.upper()} ctx=2"
        for B in args.batches:
            try:
                r = run_one(B, args.data_dir, use_v2, args.precision, args.steps)
                print(
                    f"{tag:>16s}  {r['B']:>3d}  {r['peak_alloc_gb']:>9.2f}  "
                    f"{r['peak_reserved_gb']:>12.2f}  {r['sec_per_step']:>9.3f}  "
                    f"{r['samples_per_sec']:>10.2f}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"{tag:>16s}  {B:>3d}  OOM — stopping sweep for {tag}")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                traceback.print_exc()
                print(
                    f"{tag:>16s}  {B:>3d}  ERROR {type(e).__name__}: "
                    f"{str(e)[:80]}"
                )
                torch.cuda.empty_cache()
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
