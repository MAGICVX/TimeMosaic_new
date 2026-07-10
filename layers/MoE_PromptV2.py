"""
MoE Prompt and Prefix Generators V2 — with learnable bias load balancing.

Key improvement over V1 (layers/MoE_Prompt.py):
  Uses SoftGate from MoE_Gate.py which auto-balances expert usage via
  a learnable bias, eliminating the need for explicit L1 load-balancing
  loss terms (lam_moe, lam_prefix_moe).

The interface is identical to the original MoEPromptGenerator, so it's
a drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MoE_Gate import SoftGate


# ─── MoE Prompt Generator V2 ───────────────────────────────────────────

class MoEPromptGeneratorV2(nn.Module):
    """
    Input-conditioned dynamic prompt generator via MoE with auto-balancing.

    Same architecture as MoEPromptGenerator but uses SoftGate (with
    learnable bias) instead of plain nn.Sequential gates.

    Differences from V1:
      - Gate: SoftGate with learnable bias → no explicit load-balancing loss
      - Routing weights are returned for monitoring only, not for loss
    """

    def __init__(self, d_model, num_experts, num_segs, num_latent_token,
                 hidden_size, dropout=0.1, update_bias_rate=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.num_segs = num_segs
        self.num_latent_token = num_latent_token
        self.d_model = d_model

        # Per-segment SoftGates with learnable bias (auto-balancing)
        self.gates = nn.ModuleList([
            SoftGate(
                input_dim=d_model * 2,
                num_experts=num_experts,
                update_bias_rate=update_bias_rate,
            ) for _ in range(num_segs)
        ])

        # Expert embeddings: [num_segs * num_experts, d_model]
        self.expert_embeddings = nn.Embedding(num_experts * num_segs, d_model)
        nn.init.xavier_uniform_(self.expert_embeddings.weight)

        # Per-segment transforms: mixed expert → prompt tokens
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, num_latent_token * d_model)
            ) for _ in range(num_segs)
        ])

        # Zero-init the last linear in each transform for stable training
        for transform in self.transforms:
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(transform[-1].bias)

        # Residual base prompts
        self.base_prompts = nn.Parameter(
            torch.zeros(num_segs, num_latent_token, d_model))
        nn.init.xavier_uniform_(self.base_prompts)

        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_out, extra_token):
        """
        Args:
            enc_out:     [B*C, patch_num, d_model]
            extra_token: [B*C, extra_len, d_model]

        Returns:
            prompts:         [num_segs, B*C, num_latent_token, d_model]
            routing_weights: [num_segs, B*C, num_experts]  (for monitoring)
        """
        BxC = enc_out.size(0)

        # Pool to create content representation
        patch_pooled = enc_out.mean(dim=1)        # [B*C, d_model]
        extra_pooled = extra_token.mean(dim=1)    # [B*C, d_model]
        content = torch.cat([patch_pooled, extra_pooled], dim=-1)  # [B*C, 2*d_model]

        prompts = []
        all_weights = []

        for seg in range(self.num_segs):
            # Route content to experts via learnable-bias gate
            w, _ = self.gates[seg](content)             # [B*C, num_experts]

            # Mix expert embeddings via routing weights
            expert_emb = self.expert_embeddings.weight[
                seg * self.num_experts:(seg + 1) * self.num_experts
            ]  # [num_experts, d_model]
            mixed = torch.matmul(w, expert_emb)          # [B*C, d_model]
            mixed = self.dropout(mixed)

            # Transform to prompt tokens + residual base
            delta = self.transforms[seg](mixed)          # [B*C, num_latent*d_model]
            delta = delta.view(BxC, self.num_latent_token, self.d_model)
            prompt = self.base_prompts[seg].unsqueeze(0) + delta

            prompts.append(prompt)
            all_weights.append(w)

        prompts = torch.stack(prompts, dim=0)             # [num_segs, B*C, N, D]
        routing_weights = torch.stack(all_weights, dim=0)  # [num_segs, B*C, E]

        return prompts, routing_weights


# ─── MoE Prefix Generator V2 ──────────────────────────────────────────

class MoEPrefixGeneratorV2(nn.Module):
    """
    Generates per-layer prefix K/V for attention injection via MoE.

    V2 improvements:
      - Uses SoftGate with learnable bias (auto-balancing)
      - No need for explicit prefix load-balancing loss
    """

    def __init__(self, d_model, num_experts, n_layers, prefix_len,
                 hidden_size, dropout=0.1, update_bias_rate=0.001):
        super().__init__()
        self.num_experts = num_experts
        self.n_layers = n_layers
        self.prefix_len = prefix_len
        self.d_model = d_model

        # Single SoftGate shared across all layers (auto-balancing)
        self.gate = SoftGate(
            input_dim=d_model * 2,
            num_experts=num_experts,
            update_bias_rate=update_bias_rate,
        )

        self.expert_embeddings = nn.Embedding(num_experts, d_model)
        nn.init.xavier_uniform_(self.expert_embeddings.weight)

        self.transform = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_layers * 2 * prefix_len * d_model))
        nn.init.zeros_(self.transform[-1].weight)
        nn.init.zeros_(self.transform[-1].bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, content):
        """
        content: [B*C, 2*d_model]
        Returns: [n_layers, 2, B*C, prefix_len, d_model], [B*C, E]
        """
        BxC = content.size(0)

        # Route via learnable-bias gate
        w, _ = self.gate(content)  # [B*C, num_experts]

        # Mix expert embeddings
        expert_emb = self.expert_embeddings.weight  # [E, D]
        mixed = torch.matmul(w, expert_emb)        # [B*C, D]
        mixed = self.dropout(mixed)

        # Transform to prefix tensor
        out = self.transform(mixed)
        out = out.view(BxC, self.n_layers, 2, self.prefix_len, self.d_model)
        out = out.permute(1, 2, 0, 3, 4)  # [n_layers, 2, B*C, prefix_len, D]

        return out, w
