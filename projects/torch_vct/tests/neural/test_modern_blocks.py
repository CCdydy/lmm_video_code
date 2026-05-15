import torch

from projects.torch_vct.neural.modern_blocks import (
    FlashAttnBlock,
    apply_rotary_emb,
    precompute_freqs_cis,
)


def test_apply_rotary_emb_from_real_imag_buffers():
    x = torch.randn(2, 4, 8, 16)
    freqs_cis = precompute_freqs_cis(dim=16, max_seq_len=8)

    out = apply_rotary_emb(x, freqs_cis.real, freqs_cis.imag)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_flash_attn_block_does_not_register_complex_buffers():
    block = FlashAttnBlock(dim=64, num_heads=4, max_seq_len=16)

    assert not any(torch.is_complex(buffer) for buffer in block.buffers())
    assert block.freqs_cis_real.dtype == torch.float32
    assert block.freqs_cis_imag.dtype == torch.float32

