"""Regression test: Patcher._window_partition_conv2d (unfold-based)
is numerically equivalent to the original F.conv2d(identity-kernel)
implementation it replaced (commit aa47ca1).

Why this test exists:
    The original VCT patcher used an identity-kernel conv2d that allocated
    ~600 MB per call (12288×12288 float32) and pinned it in the autograd
    graph. With six P-scenes per Vimeo septuplet at ctx=2 this was 3.6 GB
    per batch element — the dominant memory cost on a 24 GB GPU. We
    replaced the implementation with two Tensor.unfold calls (a zero-
    allocation view). This test guards against accidental regressions
    by checking that the new implementation produces:
      * bit-exact forward output (when TF32 is disabled)
      * backward gradients matching within fp32 ulp (~1e-6)
"""

import math

import pytest
import torch
import torch.nn.functional as F

from projects.torch_vct.neural.patcher import Patcher


# ---------------------------------------------------------------------------
# Reference implementation (the OLD conv2d-with-identity-kernel path)
# ---------------------------------------------------------------------------
def _window_partition_conv2d_reference(
    x: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """Verbatim copy of the pre-aa47ca1 implementation, kept as a numerical
    reference. Do not use in production — allocates ~600 MB per call for the
    VCT encoder config (C=192, patch_size=8)."""
    B, C, _, _ = x.shape
    kernel = torch.diag(x.new_ones(patch_size**2 * C)).reshape(
        C * patch_size**2, C, patch_size, patch_size
    )
    patches = F.conv2d(x, kernel, stride=stride)
    n_patches_H, n_patches_W = patches.shape[-2:]
    return (
        patches.reshape(B, C, patch_size**2, n_patches_H, n_patches_W)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
        .reshape(B * n_patches_H * n_patches_W, patch_size**2, C)
    )


def _pad_like_patcher(
    x: torch.Tensor, patch_size: int, stride: int, pad_mode: str = "reflect"
) -> torch.Tensor:
    """Replicates Patcher._pad to give both paths the same padded input."""
    missing = patch_size - stride
    assert missing % 2 == 0
    m = missing // 2
    H, W = x.shape[-2:]
    H_padded = math.ceil(H / stride) * stride
    W_padded = math.ceil(W / stride) * stride
    pad_sizes = (m, W_padded - W + m, m, H_padded - H + m)
    return F.pad(x, pad_sizes, mode=pad_mode)


# ---------------------------------------------------------------------------
# Configurations to test (the real VCT shape plus a few edge cases)
# ---------------------------------------------------------------------------
CONFIGS = [
    # (B, C, H, W, patch_size, stride)
    (2, 192, 16, 16, 8, 4),   # actual VCT encoder shape
    (4, 192, 16, 16, 8, 4),   # actual VCT, larger batch
    (1, 8, 8, 8, 4, 2),       # small CPU-friendly case
    (2, 64, 32, 32, 8, 4),    # bigger spatial
    (2, 64, 32, 32, 16, 4),   # bigger patch_size / stride ratio
]


@pytest.mark.parametrize("B,C,H,W,patch_size,stride", CONFIGS)
def test_forward_equivalence_cpu(B, C, H, W, patch_size, stride):
    """Forward output should be bit-exact on CPU (no TF32 to worry about)."""
    torch.manual_seed(0)
    x = torch.randn(B, C, H, W)
    x_padded = _pad_like_patcher(x, patch_size, stride)

    patcher = Patcher(stride=stride)
    out_new = patcher._window_partition_conv2d(x_padded, patch_size)
    out_ref = _window_partition_conv2d_reference(x_padded, patch_size, stride)

    assert torch.equal(out_new, out_ref), (
        f"Forward not bit-exact on CPU for shape "
        f"(B={B}, C={C}, H={H}, W={W}, ps={patch_size}, s={stride}); "
        f"max abs diff = {(out_new - out_ref).abs().max().item():.2e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("B,C,H,W,patch_size,stride", CONFIGS)
def test_forward_equivalence_cuda_tf32_off(B, C, H, W, patch_size, stride):
    """With TF32 disabled, CUDA fp32 forward should be bit-exact too."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.manual_seed(0)
        x = torch.randn(B, C, H, W, device="cuda")
        x_padded = _pad_like_patcher(x, patch_size, stride)

        patcher = Patcher(stride=stride)
        out_new = patcher._window_partition_conv2d(x_padded, patch_size)
        out_ref = _window_partition_conv2d_reference(x_padded, patch_size, stride)

        assert torch.equal(out_new, out_ref), (
            f"Forward not bit-exact on CUDA (TF32 off); max abs diff = "
            f"{(out_new - out_ref).abs().max().item():.2e}"
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


@pytest.mark.parametrize("B,C,H,W,patch_size,stride", CONFIGS)
def test_backward_equivalence_cpu(B, C, H, W, patch_size, stride):
    """Backward gradients should match within fp32 ulp on CPU."""
    torch.manual_seed(0)
    raw = torch.randn(B, C, H, W)

    x1 = raw.detach().clone().requires_grad_(True)
    p = Patcher(stride=stride)
    o1 = p._window_partition_conv2d(_pad_like_patcher(x1, patch_size, stride), patch_size)
    g = torch.randn_like(o1)
    o1.backward(g)
    grad_new = x1.grad.detach().clone()

    x2 = raw.detach().clone().requires_grad_(True)
    o2 = _window_partition_conv2d_reference(
        _pad_like_patcher(x2, patch_size, stride), patch_size, stride
    )
    o2.backward(g)
    grad_ref = x2.grad.detach().clone()

    # Single fp32 ulp at the magnitudes involved is ~1e-6
    assert torch.allclose(grad_new, grad_ref, atol=1e-5, rtol=1e-5), (
        f"Backward grad diff too large: max abs = "
        f"{(grad_new - grad_ref).abs().max().item():.2e}"
    )


def test_memory_is_not_quadratic():
    """Sanity check that the new path doesn't allocate the 600 MB identity
    kernel. We instantiate the encoder shape (12288×12288 worth of float32
    would be ~600 MB) and assert that peak allocation during a forward+
    backward is well below that."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to measure GPU memory")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(1, 192, 16, 16, device="cuda", requires_grad=True)
    p = Patcher(stride=4)
    out = p._window_partition_conv2d(
        _pad_like_patcher(x, patch_size=8, stride=4), patch_size=8
    )
    out.sum().backward()

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    # The old conv2d-based path peaked at ~700-800 MB for this shape due to
    # the 600 MB kernel + conv2d workspace. The unfold path should be
    # << 100 MB.
    assert peak_mb < 200.0, (
        f"Peak GPU alloc {peak_mb:.1f} MB looks like the old conv2d path "
        f"is back — expected < 200 MB for the unfold view."
    )
