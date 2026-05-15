"""
Modern LLM building blocks for VCT V2.
- RMSNorm, SwiGLU FFN, RoPE, FlashAttention block, LongCtxJointEncoder, MambaBlock
- All match the interface conventions of the existing VCT codebase.

Usage in entropy_model.py:
    from .modern_blocks import LongCtxJointEncoder

Usage in entropy_model_layers.py:
    from .modern_blocks import RMSNorm, SwiGLU_FFN, FlashAttnBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ============================================================================
# RMSNorm — replaces all LayerNorm
# ============================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA / PaLM style).

    Args:
        dim: feature dimension
        eps: epsilon for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# ============================================================================
# SwiGLU FFN — replaces 2-layer GELU MLP
# ============================================================================
class SwiGLU_FFN(nn.Module):
    """SwiGLU feed-forward network (LLaMA / PaLM style).

    Replaces the 2-layer GELU MLP with a 3-weight gated variant:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        dim: input/output dimension
        expansion: how much to expand the hidden dimension (default 4, like VCT MLP)
        dropout: dropout rate
    """

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        # Round to multiple of 128 for tensor-core efficiency
        hidden = int(2 * dim * expansion / 3)
        hidden = ((hidden + 127) // 128) * 128

        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


# ============================================================================
# RoPE — Rotary Position Embedding — replaces LearnedPosition
# ============================================================================
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tensor:
    """Precompute complex rotary embeddings for positions [0, max_seq_len).

    Args:
        dim: head dimension (must be even)
        max_seq_len: maximum sequence length to precompute for
        theta: base frequency

    Returns:
        Complex tensor of shape (max_seq_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply RoPE to query or key tensor.

    Args:
        x: (B, num_heads, seq_len, head_dim)
        freqs_cis: (max_seq_len, head_dim // 2) precomputed complex embeddings

    Returns:
        rotated tensor, same shape as x
    """
    B, n_heads, seq_len, head_dim = x.shape
    x_ = x.float().reshape(B, n_heads, seq_len, head_dim // 2, 2)
    x_complex = torch.view_as_complex(x_)
    freqs_cis = freqs_cis[:seq_len].to(x.device)
    x_rotated = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
    x_out = torch.view_as_real(x_rotated).flatten(3)
    return x_out.type_as(x)


# ============================================================================
# FlashAttnBlock — single transformer block with FlashAttention-2 + SwiGLU
# ============================================================================
class FlashAttnBlock(nn.Module):
    """Single self-attention transformer block with modern primitives.

    Pipeline:
      RMSNorm → FlashAttention-2 + RoPE → residual → RMSNorm → SwiGLU → residual

    Acts as encoder (bidirectional) or decoder (causal).

    Args:
        dim: model dimension
        num_heads: number of attention heads
        expansion: SwiGLU expansion factor
        dropout: dropout rate
        causal: if True, apply causal mask (decoder mode)
        max_seq_len: sequence length for RoPE precomputation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.causal = causal
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Precompute RoPE frequencies (non-persistent buffer)
        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # QKV in one projection (for FlashAttention compatibility)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Normalization and FFN
        self.norm_attn = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim, expansion, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, seq_len, dim)

        Returns:
            (B, seq_len, dim)
        """
        B, seq_len, D = x.shape

        # ---- Self-attention ----
        residual = x
        x_norm = self.norm_attn(x)
        qkv = self.qkv_proj(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)

        # FlashAttention-2 call — prefers (B, seq, n_heads, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        use_flash_attn = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
        try:
            if not use_flash_attn:
                raise ImportError
            from flash_attn import flash_attn_func
            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=self.causal,
            )
        except ImportError:
            # Fallback: PyTorch native scaled_dot_product_attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.causal,
            )
            attn_out = attn_out.transpose(1, 2).contiguous()

        attn_out = attn_out.reshape(B, seq_len, D)
        attn_out = self.out_proj(attn_out)
        x = residual + self.dropout(attn_out)

        # ---- SwiGLU FFN ----
        x = x + self.ffn(self.norm_ffn(x))
        return x


# ============================================================================
# LongCtxJointEncoder — V2 main module, replaces VCT's encoder_joint
# ============================================================================
class LongCtxJointEncoder(nn.Module):
    """Replaces VCT's encoder_joint (EncoderSection / Transformer encoder).

    Takes N-frame concatenated token sequences and applies hybrid attention:
    - Local branch: FlashAttention-2 on the most recent K frames (high fidelity)
    - Global branch: FlashAttention-2 on the full N-frame sequence
    - Learnable gate fuses the two outputs
    - Learned frame embedding: encodes temporal position (which frame a token
      belongs to) without polluting the 1D RoPE spatial signal.

    Interface: (B', N * frame_len, dim) → (B', N * frame_len, dim)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 16,
        num_layers: int = 4,          # VCT uses 2; V2 uses 4 for longer context
        expansion: int = 4,
        dropout: float = 0.0,
        frame_len: int = 64,          # VCT seq_len_enc = window_size_enc² = 8² = 64
        mid_window_frames: int = 16,  # local branch window (in frames)
        max_frames: int = 128,        # max number of frames; for learned frame emb
    ):
        super().__init__()
        self.frame_len = frame_len
        self.mid_window = mid_window_frames * frame_len  # in tokens

        # Learned frame embedding: one vector per frame index.
        # Added to all tokens of a frame before attention, so the model
        # knows which frame each token came from. Works alongside 1D RoPE
        # (which encodes intra-frame spatial position) to provide full
        # spatiotemporal position signal.
        self.frame_embed = nn.Embedding(max_frames, dim)

        # Local layers: only attend over the most recent mid_window tokens
        self.local_layers = nn.ModuleList([
            FlashAttnBlock(
                dim=dim,
                num_heads=num_heads,
                expansion=expansion,
                dropout=dropout,
                causal=False,
                max_seq_len=self.mid_window + 256,
            )
            for _ in range(num_layers // 2)
        ])

        # Global layers: full sequence attention
        self.global_layers = nn.ModuleList([
            FlashAttnBlock(
                dim=dim,
                num_heads=num_heads,
                expansion=expansion,
                dropout=dropout,
                causal=False,
                max_seq_len=8192,
            )
            for _ in range(num_layers - num_layers // 2)
        ])

        # Learnable gate: σ(-3) ≈ 0.047, so training starts nearly pure-local.
        # As training proceeds, if the global branch proves useful, the gate
        # will drift upward — this evolution is itself an analysable quantity.
        self.gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, seq: Tensor) -> Tensor:
        """
        Args:
            seq: (B', total_seq_len, dim) — N frames concatenated on seq-dim
                 total_seq_len = N * frame_len

        Returns:
            (B', total_seq_len, dim)
        """
        B_eff, total_len, D = seq.shape
        n_frames = total_len // self.frame_len

        # ---- Frame embedding: inject which-frame identity -------------
        # 1D RoPE inside FlashAttnBlock handles intra-frame spatial position.
        # Frame embedding handles inter-frame temporal identity.
        # Together they provide full spatiotemporal position signal.
        frame_ids = torch.arange(n_frames, device=seq.device).unsqueeze(0)  # (1, N)
        frame_ids = frame_ids.repeat_interleave(self.frame_len, dim=1)      # (1, N*64)
        seq = seq + self.frame_embed(frame_ids)

        # ---- Local processing (most recent window) ----
        local_len = min(self.mid_window, total_len)
        local_input = seq[:, -local_len:]
        local_out = local_input
        for layer in self.local_layers:
            local_out = layer(local_out)

        # Stitch: local branch output for recent tokens + original for older tokens
        local_full = torch.cat([
            seq[:, : total_len - local_len],
            local_out,
        ], dim=1)

        # ---- Global processing (full sequence) ----
        global_out = seq
        for layer in self.global_layers:
            global_out = layer(global_out)

        # ---- Gated fusion ----
        g = torch.sigmoid(self.gate)
        out = g * global_out + (1 - g) * local_full
        return out


# ============================================================================
# MambaBlock — SSM layer for experiments 4+ (N >= 64)
# ============================================================================
class MambaBlock(nn.Module):
    """Mamba SSM block usable as a transformer layer alternative.

    Requires: pip install mamba-ssm
    Falls back to Conv1d + gating if mamba_ssm is not installed.

    Args:
        dim: feature dimension
        d_state: SSM state expansion factor
        expand_factor: inner dimension expansion
    """

    def __init__(self, dim: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)

        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=4,
                expand=expand_factor,
            )
            self._have_mamba = True
        except ImportError:
            self._have_mamba = False
            self.conv = nn.Conv1d(dim, dim, kernel_size=4, padding=3, groups=dim)
            self.gate_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, seq_len, dim)

        Returns:
            (B, seq_len, dim)
        """
        residual = x
        x_norm = self.norm(x)

        if self._have_mamba:
            out = self.mamba(x_norm)
        else:
            # Fallback: conv1d along sequence + gating
            B, L, D = x_norm.shape
            x_t = x_norm.transpose(1, 2)  # (B, D, L)
            conv_out = self.conv(x_t)[:, :, :L].transpose(1, 2)
            gate = torch.sigmoid(self.gate_proj(x_norm))
            out = self.out_proj(conv_out * gate)

        return residual + out
