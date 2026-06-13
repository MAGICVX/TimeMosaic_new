import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEPromptGenerator(nn.Module):
    """
    Input-conditioned dynamic prompt generator via Mixture of Experts.
    Adapted from SEMPO's MixtrueExpertsLayer.

    Replaces static nn.Embedding prompts with content-aware routing:
    each segment has independent gates, expert pools, and transforms
    that generate prompt tokens based on the input representation.
    """

    def __init__(self, d_model, num_experts, num_segs, num_latent_token,
                 hidden_size, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.num_segs = num_segs
        self.num_latent_token = num_latent_token
        self.d_model = d_model

        # Per-segment gates: content -> expert routing weights
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_experts)
            ) for _ in range(num_segs)
        ])

        # Expert embeddings: [num_segs * num_experts, d_model]
        self.expert_embeddings = nn.Embedding(num_experts * num_segs, d_model)
        nn.init.xavier_uniform_(self.expert_embeddings.weight)

        # Per-segment transforms: mixed expert -> prompt tokens
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

        # Residual base prompts (trainable, replaces original prompt_embeddings)
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
            routing_weights: [num_segs, B*C, num_experts]
        """
        BxC = enc_out.size(0)

        # Pool to create content representation
        patch_pooled = enc_out.mean(dim=1)        # [B*C, d_model]
        extra_pooled = extra_token.mean(dim=1)    # [B*C, d_model]
        content = torch.cat([patch_pooled, extra_pooled], dim=-1)  # [B*C, 2*d_model]

        prompts = []
        all_weights = []

        for seg in range(self.num_segs):
            # Route content to experts
            logits = self.gates[seg](content)            # [B*C, num_experts]
            w = F.softmax(logits, dim=-1)                # [B*C, num_experts]

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
