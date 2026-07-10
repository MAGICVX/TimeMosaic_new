"""
MoE Gate with Learnable Load-Balancing Bias.

Adapted from Kairos (refer/Kairos/tsfm/model/kairos/moe.py).

Key improvement over the original Fusion approach:
  Instead of adding an explicit L1 load-balancing loss term (which creates
  gradient conflict with the main prediction loss), the gate maintains a
  learnable bias that auto-adjusts during training to match a target
  distribution across experts.

How it works:
  1. scores = softmax(x @ W + bias)
  2. Top-k selection for routing
  3. After forward: bias += rate * (target - actual_usage)
     (no gradient through this update — it's a stateful adjustment)

This eliminates the need for lam_moe and lam_prefix_moe hyperparameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableBiasGate(nn.Module):
    """
    Gate with auto-updating bias for expert load balancing.

    The bias is updated each forward pass (no gradient) to nudge expert
    usage toward the target distribution. This replaces the explicit
    L1 regularization loss used in the original Fusion.
    """
    def __init__(self, input_dim: int, num_experts: int,
                 topk: int = 1, update_bias_rate: float = 0.001,
                 target_dist=None, route_scale: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.update_bias_rate = update_bias_rate
        self.route_scale = route_scale

        self.weight = nn.Parameter(torch.empty(num_experts, input_dim))
        self.register_buffer('bias', torch.zeros(num_experts))

        if target_dist is not None:
            if isinstance(target_dist, float):
                target_dist = [target_dist]
            assert abs(sum(target_dist) - 1.0) < 1e-10, \
                f"Target distribution must sum to 1.0, got {sum(target_dist)}"
            self.target_dist = torch.tensor(target_dist)
        else:
            self.target_dist = torch.ones(num_experts) / num_experts

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, input_dim]  input features for routing
        Returns:
            weights: [B, topk]  normalized routing weights
            indices: [B, topk]  expert indices
        """
        # Compute routing scores with learnable bias
        scores = F.linear(torch.nan_to_num(x, nan=0.0), self.weight)
        scores = scores + self.bias
        scores = scores.softmax(dim=-1, dtype=torch.float32)

        # Top-k selection
        original_scores = scores
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale

        # Update bias for load balancing (no gradient)
        if self.training:
            self._update_bias(indices, weights)

        return weights.type_as(x), indices

    def _update_bias(self, indices, weights):
        """Auto-adjust bias to match target distribution."""
        flatten_indices = indices.view(-1)
        flatten_weights = weights.view(-1)
        expert_weights_sum = torch.bincount(
            flatten_indices, weights=flatten_weights,
            minlength=self.num_experts).float()
        total_weights = expert_weights_sum.sum()

        target_dist = self.target_dist.to(
            device=expert_weights_sum.device, dtype=expert_weights_sum.dtype)
        expected_weights_sum = target_dist * total_weights
        load_error = expected_weights_sum - expert_weights_sum

        with torch.no_grad():
            self.bias += self.update_bias_rate * (load_error / (total_weights + 1e-8))

    def get_current_bias(self):
        """For logging/monitoring."""
        return self.bias.detach().clone()


class SoftGate(nn.Module):
    """
    Soft gating variant: uses full softmax (no top-k sparsity).
    Includes learnable bias for load balancing. Suitable for prompt/prefi
    generation where we want all experts to contribute.
    """
    def __init__(self, input_dim: int, num_experts: int,
                 update_bias_rate: float = 0.001,
                 target_dist=None):
        super().__init__()
        self.num_experts = num_experts
        self.update_bias_rate = update_bias_rate

        self.weight = nn.Parameter(torch.empty(num_experts, input_dim))
        self.register_buffer('bias', torch.zeros(num_experts))

        if target_dist is not None:
            if isinstance(target_dist, float):
                target_dist = [target_dist]
            self.target_dist = torch.tensor(target_dist)
        else:
            self.target_dist = torch.ones(num_experts) / num_experts

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, input_dim]
        Returns:
            weights: [B, num_experts]  softmax routing weights (full, not top-k)
        """
        scores = F.linear(torch.nan_to_num(x, nan=0.0), self.weight)
        scores = scores + self.bias
        w = scores.softmax(dim=-1, dtype=torch.float32)

        # Update bias (no gradient)
        if self.training:
            self._update_bias(w)

        return w.type_as(x), None  # no indices needed for soft gating

    def _update_bias(self, weights):
        """Auto-adjust bias. weights: [B, num_experts]"""
        expert_weights_sum = weights.sum(dim=0)  # [num_experts]
        total_weights = expert_weights_sum.sum()

        target_dist = self.target_dist.to(
            device=expert_weights_sum.device, dtype=expert_weights_sum.dtype)
        expected_weights_sum = target_dist * total_weights
        load_error = expected_weights_sum - expert_weights_sum

        with torch.no_grad():
            self.bias += self.update_bias_rate * (load_error / (total_weights + 1e-8))
