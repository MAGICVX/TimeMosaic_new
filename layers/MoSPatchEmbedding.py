"""
MoS Patch Embedding — Kairos-style hierarchical patching for TimeMosaic.

Replaces AdaptivePatchEmbedding's Gumbel-Softmax soft selection with
hard MoE routing + hierarchical splitting. Each region is routed to a
granularity level (0..levels-1), then hierarchically split to create
variable-size patches. A per-granularity Linear layer embeds each patch.

Key improvements over AdaptivePatchEmbedding:
  1. Hard routing with learnable bias (no auxiliary loss needed)
  2. Hierarchical splitting produces variable patch counts per sample
  3. Per-granularity embeddings (different weights for 32/16/8-size patches)
  4. No computation wasted on unselected granularities
  5. Fixed-size output via deterministic padding (R * max_patches_per_region)
  6. Optional routing reuse for multi-view consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.MoE_Gate import LearnableBiasGate
from layers.MoSPatchUtils import (
    _divide_patches,
    _update_parent_mapping,
    _update_position_mapping,
)


class MoSPatchEmbedding(nn.Module):
    """
    MoS hierarchical patch tokenizer with fixed-size output.

    Workflow:
      1. Input [B*C, L] → divide into R regions [B*C, R, max_patch_len]
      2. MoS Gate routes each region → granularity level k in {0..levels-1}
      3. Hierarchical splitting: levels-1 rounds; round i splits regions
         where expert > i into two child patches
      4. Pad to fixed max_patches = R * max_patches_per_region
      5. Embed each leaf patch with per-granularity Linear layer
      6. Return padded embeddings (zero-padded patches → zero embeddings)

    Args:
        max_patch_len: initial (coarsest) patch length (e.g. 32)
        levels: number of granularity levels (e.g. 3 -> sizes: 32, 16, 8)
        d_model: embedding dimension
        seq_len: input sequence length (for computing R)
        update_bias_rate: learning rate for MoS gate bias auto-update
        dropout: dropout after embedding
        training: whether in training mode
    """

    def __init__(
        self,
        max_patch_len: int = 32,
        levels: int = 3,
        d_model: int = 512,
        seq_len: int = 96,
        update_bias_rate: float = 0.001,
        dropout: float = 0.1,
        training: bool = True,
    ):
        super().__init__()
        self.max_patch_len = max_patch_len
        self.levels = levels
        self.d_model = d_model
        self.training = training

        # Region count
        self.R = seq_len // max_patch_len

        # Per-level effective patch sizes
        self.patch_sizes = [
            max(1, max_patch_len // (2 ** i)) for i in range(levels)
        ]
        self.min_patch_len = self.patch_sizes[-1]

        # Maximum patches per region after full splitting
        self.max_patches_per_region = max_patch_len // self.min_patch_len

        # Fixed total patch count (padded to this size)
        self.max_total_patches = self.R * self.max_patches_per_region

        # MoS routing gate
        self.gate = LearnableBiasGate(
            input_dim=max_patch_len,
            num_experts=levels,
            topk=1,
            update_bias_rate=update_bias_rate,
        )

        # Per-granularity embeddings
        self.embeddings = nn.ModuleList([
            nn.Linear(max_patch_len, d_model, bias=False)
            for _ in range(levels)
        ])

        self.dropout = nn.Dropout(dropout)

        # Monitoring: routing distribution
        self.register_buffer('latest_cls_soft', torch.zeros(levels))

    def forward(self, x, reuse_expert_indices=None):
        """
        Args:
            x: [BxC, L]  flattened batch-channel input
            reuse_expert_indices: [BxC * R] optional pre-computed routing indices
                When provided, skips gate computation and reuses these decisions.
                Used for multi-view consistency (raw view routes, freq views reuse).

        Returns:
            patches:        [BxC, max_total_patches, d_model]  padded embeddings
            C:              int  always 1 after flattening
            cls_pred:       [BxC * R]  expert indices per region (for monitoring)
            expert_indices: [BxC * R]  raw gate indices (for reuse by other views)
        """
        BxC, L = x.shape
        R = self.R

        # ── 1. Divide into regions ──
        regions = x.view(BxC, R, self.max_patch_len).contiguous()

        # ── 2. Routing ──
        if reuse_expert_indices is not None:
            # Reuse routing from another view
            gate_indices = reuse_expert_indices.view(BxC, R, 1)
            gate_weights = torch.ones_like(gate_indices, dtype=torch.float32)
        else:
            # Fresh routing via MoS Gate
            regions_flat = regions.view(BxC * R, self.max_patch_len)
            gate_weights, gate_indices = self.gate(regions_flat)
            gate_weights = gate_weights.view(BxC, R, 1)
            gate_indices = gate_indices.view(BxC, R, 1)

        index_history = gate_indices.clone()  # save for return

        # ── 3. Hierarchical splitting ──
        size = torch.full((BxC, R), self.max_patch_len,
                          device=x.device, dtype=torch.long)
        x_patches = regions
        weights = gate_weights
        current_indices = gate_indices

        parent_mapping = torch.arange(R, device=x.device).unsqueeze(0).expand(BxC, -1)
        position_mapping = torch.zeros((BxC, R, 2), dtype=torch.long, device=x.device)
        position_mapping[:, :, 0] = 0
        position_mapping[:, :, 1] = self.max_patch_len

        for level in range(self.levels - 1):
            to_divide = (current_indices.squeeze(-1) > level)
            if not to_divide.any():
                continue
            div_counts = to_divide.sum(dim=1)
            x_patches, size, weights, current_indices = _divide_patches(
                x_patches, size, to_divide, weights, current_indices)
            parent_mapping = _update_parent_mapping(
                parent_mapping, to_divide, div_counts, x.device)
            position_mapping = _update_position_mapping(
                position_mapping, to_divide, div_counts, x.device)

        # ── 4. Pad to fixed size ──
        BxC_2, cur_patches, _ = x_patches.shape
        if cur_patches < self.max_total_patches:
            pad_len = self.max_total_patches - cur_patches
            x_patches = F.pad(x_patches, (0, 0, 0, pad_len))
            size = F.pad(size, (0, pad_len))
            current_indices = F.pad(current_indices, (0, 0, 0, pad_len),
                                    value=-1)
        elif cur_patches > self.max_total_patches:
            x_patches = x_patches[:, :self.max_total_patches, :]
            size = size[:, :self.max_total_patches]
            current_indices = current_indices[:, :self.max_total_patches, :]

        # ── 5. Embed per granularity ──
        leaf_indices = current_indices.squeeze(-1).clamp(0, self.levels - 1)

        embedded = torch.zeros(BxC_2, self.max_total_patches, self.d_model,
                               device=x.device, dtype=x_patches.dtype)

        for g in range(self.levels):
            mask_g = (leaf_indices == g) & (size > 0)
            if mask_g.any():
                patches_g = x_patches[mask_g]
                embedded[mask_g] = self.embeddings[g](patches_g)

        embedded = self.dropout(embedded)

        # ── 6. Monitoring ──
        cls_pred = index_history.view(-1)

        if self.training:
            with torch.no_grad():
                valid = (leaf_indices >= 0) & (size > 0)
                if valid.any():
                    usage = F.one_hot(
                        leaf_indices[valid].clamp(0, self.levels - 1),
                        num_classes=self.levels).float()
                    self.latest_cls_soft = usage.mean(dim=0)

        return embedded, 1, cls_pred, index_history.view(-1)
