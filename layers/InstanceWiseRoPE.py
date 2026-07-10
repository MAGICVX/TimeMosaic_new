"""
Instance-Wise Rotary Position Embedding (DRoPE).

Adapted from Kairos (refer/Kairos/tsfm/model/kairos/tunable_rope_utils.py).
Replaces static sinusoidal positional encoding with instance-conditioned
rotary embeddings: each time series gets its own RoPE frequency scaling
based on its FFT spectral features.

Key insight: Fusion already computes FFT for spectral decomposition, so
s_features are essentially a free by-product.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── FFT Feature Extractor ──────────────────────────────────────────────

class FFTFeatureExtractor(nn.Module):
    """
    Extracts spectral features from raw time series for DRoPE conditioning.
    Mirrors Kairos's fft_process: FFT → amplitude → pad/trim → LayerNorm.
    """
    def __init__(self, seq_len: int, feature_dim: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.fft_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        """
        Args:
            x: [B, T, C]  raw or normalized time series
        Returns:
            s_features: [B, feature_dim]  per-instance spectral features
        """
        B, T, C = x.shape
        # FFT along time dim, take amplitude
        x_fft = torch.fft.rfft(x, dim=1).abs()  # [B, freq_len, C]
        # Average over channels
        s_features = x_fft.mean(dim=-1)  # [B, freq_len]
        # Pad or trim to feature_dim
        if s_features.shape[1] >= self.feature_dim:
            s_features = s_features[:, :self.feature_dim]
        else:
            pad_len = self.feature_dim - s_features.shape[1]
            s_features = F.pad(s_features, (0, pad_len))
        s_features = self.fft_norm(s_features)
        return s_features


# ─── Instance-Wise Parameter Network ────────────────────────────────────

class InstanceWiseParamNet(nn.Module):
    """
    Small MLP that predicts per-instance gamma (scale) and beta (shift)
    for the RoPE log-frequencies.

    Architecture: input → 128 → 64 → 2*theta_dim
    Final layer bias is initialized so gamma=1, beta=0 (identity at start).
    """
    def __init__(self, input_feature_dim: int, theta_dim: int):
        super().__init__()
        self.theta_dim = theta_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.theta_dim),
        )
        self._init_final_layer()

    def _init_final_layer(self):
        final_layer = self.mlp[-1]
        if final_layer.bias is not None:
            with torch.no_grad():
                # gamma (first half) → 1.0
                final_layer.bias[:self.theta_dim] = 1.0
                # beta (second half) → 0.0
                final_layer.bias[self.theta_dim:] = 0.0

    def forward(self, s_features: torch.Tensor):
        """
        Args:
            s_features: [B, feature_dim]
        Returns:
            gamma: [B, theta_dim]  frequency scaling factors
            beta:  [B, theta_dim]  frequency shift factors
        """
        params = self.mlp(s_features)
        gamma = params[:, :self.theta_dim]
        beta = params[:, self.theta_dim:]
        return gamma, beta


# ─── Instance-Wise Rotary Embedding ─────────────────────────────────────

class InstanceWiseRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding whose frequencies are modulated per-instance.

    Standard RoPE:  freqs = 1 / (base^(2i/d))
    DRoPE:          freqs = exp(gamma * log(freqs) + beta)

    This allows the model to adapt its positional encoding to the dominant
    periodicities of each individual time series.
    """
    def __init__(self, dim: int, input_feature_dim: int,
                 base: float = 10000.0, init: str = "exp",
                 min_period: float = 0.01, max_period: float = 1000.0):
        super().__init__()
        if init == 'exp':
            theta = self._get_exp_period(min_period, max_period, dim)
        else:
            theta = 1.0 / (base ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim))

        self.register_buffer('log_freqs', torch.log(theta))
        self.dim = dim
        self.param_net = InstanceWiseParamNet(input_feature_dim, theta_dim=dim // 2)

    @staticmethod
    def _get_exp_period(min_period, max_period, dim):
        i = torch.arange(0, dim, 2)[:dim // 2]
        max_theta = 2 * np.pi / min_period
        min_theta = 2 * np.pi / max_period
        alpha = np.log(max_theta / min_theta) / (dim - 2)
        thetas = max_theta * np.exp(-alpha * i)
        return thetas

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, s_features: torch.Tensor):
        """
        Args:
            xq: [B, H, L, D]  query states (post-projection, pre-attention)
            xk: [B, H, S, D]  key states
            s_features: [B, feature_dim]  per-sample spectral features
        Returns:
            xq_out, xk_out: same shapes, rotated
        """
        B, H, L, D = xq.shape
        _, _, S, _ = xk.shape

        gamma, beta = self.param_net(s_features)  # [B, dim//2]

        # Scale and shift log-frequencies per instance
        # gamma: [B, dim//2], log_freqs: [dim//2]
        scaled_log_freqs = gamma * self.log_freqs.unsqueeze(0) + beta  # [B, dim//2]
        scaled_freqs = torch.exp(scaled_log_freqs)  # [B, dim//2]

        def rotate(x, Lx):
            # x: [B, H, Lx, D] → flatten batch-head
            x = x.reshape(-1, Lx, D)  # [B*H, Lx, D]
            t = torch.arange(Lx, device=x.device).float()
            # freqs: [B*H, Lx, dim//2]
            freqs = torch.einsum('l,bd->bld', t, scaled_freqs.repeat_interleave(H, dim=0))
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            # Convert to complex and rotate
            x_ = x.float().reshape(*x.shape[:-1], -1, 2)
            x_ = torch.view_as_complex(x_)
            x_out = torch.view_as_real(x_ * freqs_cis).flatten(2)
            return x_out.reshape(B, H, Lx, D).type_as(xq)

        xq_out = rotate(xq, L)
        xk_out = rotate(xk, S)
        return xq_out, xk_out


# ─── RoPE-Aware Attention Layer ───────────────────────────────────────

class RoPEFullAttention(nn.Module):
    """
    FullAttention variant that applies DRoPE to q, k before computing scores.
    Wraps the standard scaled dot-product attention with rotary embeddings.
    """
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1,
                 output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None,
                tau=None, delta=None, rope=None, s_features=None):
        """
        Args:
            queries: [B, L, H, E]
            keys:    [B, S, H, E]
            values:  [B, S, H, D]
            rope:    InstanceWiseRotaryEmbedding (or None)
            s_features: [B, feature_dim]  (or None)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / (E ** 0.5)

        # Apply DRoPE if provided
        # NOTE: RoPE expects [B, H, L, D] but attention uses [B, L, H, D]
        if rope is not None and s_features is not None:
            queries = queries.permute(0, 2, 1, 3)  # [B, H, L, E]
            keys = keys.permute(0, 2, 1, 3)        # [B, H, S, E]
            queries, keys = rope(queries, keys, s_features)
            queries = queries.permute(0, 2, 1, 3)  # back to [B, L, H, E]
            keys = keys.permute(0, 2, 1, 3)        # back to [B, S, H, E]

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            from utils.masking import TriangularCausalMask
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, float('-inf'))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class RoPEAttentionLayer(nn.Module):
    """
    AttentionLayer that applies DRoPE (Instance-Wise Rotary Embedding)
    between projection and attention computation.

    Extends the standard AttentionLayer pattern to accept rope and s_features.
    """
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None,
                 attention_dropout=0.1, output_attention=False):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = RoPEFullAttention(
            mask_flag=False, scale=None, attention_dropout=attention_dropout,
            output_attention=output_attention)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None,
                tau=None, delta=None, rope=None, s_features=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask=attn_mask,
            tau=tau, delta=delta, rope=rope, s_features=s_features)

        out = out.view(B, L, -1)
        return self.out_projection(out), attn
