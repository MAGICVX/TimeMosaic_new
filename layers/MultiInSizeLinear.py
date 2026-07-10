"""
MultiInSizeLinear + MultiInResidualBlock — multi-granularity embedding layers.

Adapted from Kairos (refer/Kairos/tsfm/model/kairos/layers.py).

These layers handle patches of different sizes by maintaining separate
weight matrices for each granularity level. A mask zeros out irrelevant
input positions for each level.

Key class:
    MultiInSizeLinear: linear layer with per-granularity weight matrices
    MultiInResidualBlock: residual block wrapper with multi-size support
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from einops import einsum


def size_to_mask(max_size: int, sizes: torch.Tensor) -> torch.Tensor:
    """Create boolean mask of shape [len(sizes), max_size]."""
    mask = torch.arange(max_size, device=sizes.device)
    return torch.lt(mask, sizes.unsqueeze(-1))


# ─── MultiInSizeLinear ──────────────────────────────────────────────────

class MultiInSizeLinear(nn.Module):
    """
    Linear layer that handles multiple input feature sizes.

    Maintains independent weight matrices for each granularity level.
    During inference with expert_indices, only the weight for the
    assigned granularity is used per patch.

    Args:
        in_features_ls: list of feature sizes per granularity level
            e.g. [32, 16, 8] for 3 levels
        out_features: output dimension
        bias: whether to include bias
        dtype: optional data type for weights
    """
    def __init__(
        self,
        in_features_ls: Tuple[int, ...],
        out_features: int,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features_ls = in_features_ls
        self.out_features = out_features
        max_feat = max(in_features_ls)

        self.weight = nn.Parameter(
            torch.empty((len(in_features_ls), out_features, max_feat * 2), dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty((len(in_features_ls), out_features), dtype=dtype))
        else:
            self.register_parameter("bias", None)

        # Build mask: [num_feat_sizes, 1, max_feat * 2]
        mask_raw = torch.cat((
            size_to_mask(max_feat, torch.as_tensor(in_features_ls)),
            size_to_mask(max_feat, torch.as_tensor(in_features_ls)),
        ), dim=-1)  # [num_feat_sizes, max_feat * 2]
        mask_raw = mask_raw.unsqueeze(1)  # [num_feat_sizes, 1, max_feat * 2]
        self.register_buffer("mask", mask_raw, persistent=False)
        self.register_buffer(
            "in_features_buffer",
            torch.tensor(in_features_ls),
            persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        for idx, feat_size in enumerate(self.in_features_ls):
            nn.init.kaiming_uniform_(self.weight[idx, :, :feat_size], a=math.sqrt(5))
            nn.init.zeros_(self.weight[idx, :, feat_size:])
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.weight[idx, :, :feat_size])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[idx], -bound, bound)

    def forward(
        self,
        x: torch.Tensor,                            # [*batch, max_feat * 2]
        in_feat_size: Optional[torch.Tensor] = None,  # [*batch] not used with expert_indices
        expert_weights: Optional[torch.Tensor] = None, # [*batch, n_experts]
        expert_indices: Optional[torch.Tensor] = None, # [*batch, n_experts]
        x_final: Optional[torch.Tensor] = None,        # [max_granularity, total, max_feat*2]
    ) -> torch.Tensor:
        """
        Forward with optional MoS expert routing.
        If expert_indices is provided, uses x_final for per-granularity
        feature rearrangement.
        """
        if expert_indices is not None and x_final is not None:
            return self._forward_mos(x, expert_weights, expert_indices, x_final)
        return self._forward_vanilla(x, in_feat_size, expert_weights)

    def _forward_mos(self, x, expert_weights, expert_indices, x_final):
        """MoS path: use expert routing + x_final rearranged features."""
        batch_shape = x.shape[:-1]
        total_patches = x.view(-1, x.size(-1)).size(0)

        out = torch.zeros((total_patches, self.out_features), device=x.device, dtype=x.dtype)
        expert_weights = expert_weights.view(-1, expert_weights.size(-1))
        expert_indices = expert_indices.view(-1, expert_indices.size(-1))
        n_real_experts = len(self.in_features_ls)

        for k in range(expert_indices.size(1)):
            indices_k = expert_indices[:, k]
            weights_k = expert_weights[:, k]

            is_real_expert_mask = (indices_k < n_real_experts)
            if not is_real_expert_mask.any():
                continue

            for feat_idx, feat_size in enumerate(self.in_features_ls):
                mask = (indices_k == feat_idx) & is_real_expert_mask
                if not mask.any():
                    continue

                weight = self.weight[feat_idx] * self.mask[feat_idx]
                bias = self.bias[feat_idx] if self.bias is not None else 0
                x_masked = x_final[feat_idx][mask]

                expert_out = einsum(weight, x_masked, "out inp, ... inp -> ... out") + bias

                # Normalize weights among real experts only
                real_expert_weights_sum = (
                    expert_weights * (expert_indices < n_real_experts).float()
                ).sum(dim=-1, keepdim=True)
                real_expert_weights_sum[real_expert_weights_sum == 0] = 1.0

                weights_k_norm = weights_k / real_expert_weights_sum.view(-1)
                weighted_out = expert_out * weights_k_norm[mask].unsqueeze(-1)
                out[mask] += weighted_out

        return out.view(*batch_shape, self.out_features)

    def _forward_vanilla(self, x, in_feat_size, expert_weights):
        """Vanilla path: use in_feat_size to select weight matrix."""
        out = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for idx, feat_size in enumerate(self.in_features_ls):
            weight = self.weight[idx] * self.mask[idx]
            bias = self.bias[idx] if self.bias is not None else 0
            out = out + (
                torch.eq(in_feat_size, feat_size).unsqueeze(-1)
                * (einsum(weight, x, "out inp, ... inp -> ... out") + bias))
        if expert_weights is not None:
            out = expert_weights.unsqueeze(-1) * out
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features_ls={self.in_features_ls}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}")


# ─── MultiInResidualBlock ────────────────────────────────────────────────

class MultiInResidualBlock(nn.Module):
    """
    Residual block with multi-size input support.

    Uses MultiInSizeLinear for both the hidden and residual projections,
    allowing each granularity level to have its own transformation.

    Args:
        in_dim_ls: tuple of input sizes per granularity level
        h_dim: hidden dimension
        out_dim: output dimension
        act_fn_name: activation function ('relu' or 'gelu')
        dropout_p: dropout probability
        use_layer_norm: whether to apply LayerNorm after residual
    """
    def __init__(
        self,
        in_dim_ls: Tuple[int, ...],
        h_dim: int,
        out_dim: int,
        act_fn_name: str = "relu",
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = MultiInSizeLinear(in_dim_ls, h_dim)
        self.act = nn.ReLU() if act_fn_name == "relu" else nn.GELU()
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = MultiInSizeLinear(in_dim_ls, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        in_feat_size: Optional[torch.Tensor] = None,
        expert_weights: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        x_final: Optional[torch.Tensor] = None,
    ):
        hid = self.act(self.hidden_layer(x, in_feat_size, expert_weights, expert_indices, x_final))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x, in_feat_size, expert_weights, expert_indices, x_final)

        out = out + res
        if self.use_layer_norm:
            return self.layer_norm(out)
        return out
