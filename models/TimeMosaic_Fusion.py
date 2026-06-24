"""
TimeMosaic × SEMPO Full Fusion Architecture.

Integrates:
  - Spectral decomposition (SEMPO)
  - Adaptive multi-granularity patching (TimeMosaic)
  - MoE dynamic prompt (Scheme B)
  - MoE prefix for attention K/V injection (SEMPO-style)
  - Reconstruction auxiliary task (SEMPO)
"""

from layers.Transformer_EncDec import Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.revin import RevIN
from layers.MoE_Prompt import MoEPromptGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── EncoderLayer with optional prefix K/V injection ────────────────

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", num_latent_token=0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.num_latent_token = num_latent_token

    def forward(self, x, attn_mask=None, tau=None, delta=None, prefix_kv=None):
        q = self.mask_last_tokens(x)
        if prefix_kv is not None:
            k = torch.cat([prefix_kv[0], x], dim=1)
            v = torch.cat([prefix_kv[1], x], dim=1)
        else:
            k, v = x, x
        new_x, attn = self.attention(
            q, k, v, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

    def mask_last_tokens(self, x):
        x_masked = x.clone()
        if self.num_latent_token > 0:
            x_masked[:, :self.num_latent_token, :] = 0
        return x_masked


# ─── Prefix-aware Encoder wrapper ────────────────────────────────────

class PrefixEncoder(nn.Module):
    """Encoder that passes per-layer prefix K/V to each EncoderLayer."""
    def __init__(self, attn_layers, norm_layer):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, prefix_kv=None):
        # x: [B, L, D], prefix_kv: [n_layers, 2, B, P, D] or None
        attn = None
        for i, layer in enumerate(self.attn_layers):
            pkv = prefix_kv[i] if prefix_kv is not None else None
            x, attn = layer(x, attn_mask=attn_mask, prefix_kv=pkv)
        if self.norm is not None:
            x = self.norm(x)
        return x, attn


# ─── Positional Embedding ────────────────────────────────────────────

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# ─── Adaptive Patch Embedding (TimeMosaic core) ──────────────────────

class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len_list, mode='fixed', dropout=0.0,
                 seq_len=96, in_channels=1, training=True):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.mode = mode
        self.max_patch_len = max(patch_len_list)
        self.min_patch_len = min(patch_len_list)
        self.region_num = seq_len // self.max_patch_len
        self.d_model = d_model
        self.in_channels = in_channels
        self.training = training
        self.register_buffer('target_ratio',
                             torch.ones(len(patch_len_list)) / len(patch_len_list))
        self.region_cls = nn.Sequential(
            nn.Linear(self.max_patch_len, 64),
            nn.ReLU(),
            nn.Linear(64, len(patch_len_list)))
        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model, bias=False)
            for patch_len in patch_len_list])
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B * C, self.region_num, self.max_patch_len)
        all_patches, cls_pred_list, cls_soft_list = [], [], []
        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]
            cls_logits = self.region_cls(region)
            if self.training:
                cls_soft = F.gumbel_softmax(cls_logits, tau=0.5, hard=True, dim=-1)
            else:
                cls_pred = torch.argmax(cls_logits, dim=-1)
                cls_soft = F.one_hot(cls_pred, num_classes=len(self.patch_len_list)).float()
            cls_soft_list.append(cls_soft)
            cls_pred_list.append(cls_soft.argmax(dim=-1))
            patch_emb_list = []
            for idx, patch_len in enumerate(self.patch_len_list):
                patches = region.unfold(-1, patch_len, patch_len)
                if self.mode == 'fixed':
                    target_patch_num = self.max_patch_len // self.min_patch_len
                    repeat = target_patch_num - patches.size(1)
                    if repeat > 0:
                        patches = patches.repeat_interleave(
                            repeat + 1, dim=1)[:, :target_patch_num, :]
                patch_emb_list.append(self.embeddings[idx](patches))
            patch_emb_stack = torch.stack(patch_emb_list, dim=0)
            cls_soft_trans = cls_soft.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            region_patches_sorted = (patch_emb_stack * cls_soft_trans).sum(dim=0)
            all_patches.append(region_patches_sorted)
        x_patch = torch.cat(all_patches, dim=1)
        x_patch += self.position_embedding(x_patch)
        x_patch = self.dropout(x_patch)
        all_cls_pred = torch.cat(cls_pred_list, dim=0)
        self.latest_cls_soft = torch.cat(cls_soft_list, dim=0)
        return x_patch, C, all_cls_pred


# ─── MoE Prefix Generator ────────────────────────────────────────────

class MoEPrefixGenerator(nn.Module):
    """Generates per-layer prefix K/V for attention injection via MoE."""

    def __init__(self, d_model, num_experts, n_layers, prefix_len,
                 hidden_size, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.n_layers = n_layers
        self.prefix_len = prefix_len
        self.d_model = d_model

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts))
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
        """content: [B*C, 2*d_model]
        Returns: [n_layers, 2, B*C, prefix_len, d_model]"""
        BxC = content.size(0)
        logits = self.gate(content)
        w = F.softmax(logits, dim=-1)
        expert_emb = self.expert_embeddings.weight  # [E, D]
        mixed = torch.matmul(w, expert_emb)
        mixed = self.dropout(mixed)
        out = self.transform(mixed)
        out = out.view(BxC, self.n_layers, 2, self.prefix_len, self.d_model)
        out = out.permute(1, 2, 0, 3, 4)  # [n_layers, 2, B*C, prefix_len, D]
        return out, w


# ─── Pretrain / Reconstruction Head (from SEMPO) ─────────────────────

class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        # x: [B, n_vars, d_model, patch_num]
        x = x.transpose(2, 3)           # [B, n_vars, patch_num, d_model]
        x = self.linear(self.dropout(x))  # [B, n_vars, patch_num, patch_len]
        x = x.permute(0, 2, 1, 3)       # [B, patch_num, n_vars, patch_len]
        return x


# ─── Utilities ───────────────────────────────────────────────────────

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# ═══════════════════════════════════════════════════════════════════════
#  FUSION MODEL
# ═══════════════════════════════════════════════════════════════════════

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.training = configs.is_training

        self.patch_len_list = eval(configs.patch_len_list)
        seg_len_map = {96: configs.pre96, 192: configs.pre192,
                       336: configs.pre336, 720: configs.pre720}
        self.seg_len = seg_len_map.get(configs.pred_len, configs.pre12)
        self.num_segs = self.pred_len // self.seg_len
        self.channel = configs.channel

        # ── Spectral decomposition (SEMPO) ──
        self.freq_num = getattr(configs, 'freq_num', 4)
        if self.freq_num > 0:
            self.freq_len = configs.seq_len // 2 + 1
            self.theta = nn.Parameter(torch.rand(1))
            self.tau_main = nn.Parameter(torch.rand(self.freq_num) * (configs.seq_len // 2 + 1))
            self.mu_main = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
            self.tau_res = nn.Parameter(torch.rand(self.freq_num) * (configs.seq_len // 2 + 1))
            self.mu_res = nn.Parameter(torch.bernoulli(torch.full((self.freq_num, self.freq_len), 0.5)))
        self.num_views = self.freq_num + 1  # raw + freq_num freq views

        # ── Adaptive patching (shared across views) ──
        self.patch_embedding = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            patch_len_list=self.patch_len_list,
            mode='fixed', dropout=configs.dropout,
            seq_len=configs.seq_len,
            in_channels=configs.enc_in,
            training=configs.is_training)

        # ── Cross-view aggregation ──
        self.view_projection = nn.Linear(self.num_views * configs.d_model,
                                         configs.d_model)

        # ── MoE dynamic prompt ──
        self.num_latent_token = configs.num_latent_token
        self.num_moe_experts = getattr(configs, 'num_moe_experts', 8)
        self.prompt_moe = MoEPromptGenerator(
            d_model=configs.d_model,
            num_experts=self.num_moe_experts,
            num_segs=self.num_segs,
            num_latent_token=configs.num_latent_token,
            hidden_size=configs.d_model,
            dropout=configs.dropout)

        # ── MoE prefix for attention K/V ──
        self.num_moe_prefix_experts = getattr(configs, 'num_moe_prefix_experts', 4)
        self.prefix_len = getattr(configs, 'prefix_len', 4)
        self.use_prefix = getattr(configs, 'use_prefix', True)
        if self.use_prefix:
            self.prefix_moe = MoEPrefixGenerator(
                d_model=configs.d_model,
                num_experts=self.num_moe_prefix_experts,
                n_layers=configs.e_layers,
                prefix_len=self.prefix_len,
                hidden_size=configs.d_model,
                dropout=configs.dropout)
        else:
            self.prefix_moe = None

        # ── Prefix-aware Encoder ──
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor,
                                  attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation)
            for _ in range(configs.e_layers)]
        self.encoder = PrefixEncoder(
            encoder_layers,
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)))

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq,
            configs.dropout)

        # ── Prediction heads ──
        self.head_nf = configs.d_model * int(
            (configs.seq_len - patch_len) / stride + 2)
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.heads = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf, self.seg_len,
                        head_dropout=configs.dropout)
            for _ in range(self.num_segs)])

        # ── Reconstruction head (auxiliary task) ──
        self.use_reconstruct = getattr(configs, 'use_reconstruct', True)
        if self.use_reconstruct:
            self.reconstruct_head = FlattenHead(
                configs.enc_in, self.head_nf, configs.seq_len,
                head_dropout=configs.dropout)
        else:
            self.reconstruct_head = None

        # ── Masking (for MIM-style pretraining) ──
        self.revin = False
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.mask_ratio = getattr(configs, "mask_ratio", 0)
        self.mask_ratio_patch = getattr(configs, "mask_ratio_patch", 0)
        if self.mask_ratio > 0:
            self.mask_ratio_patch = 0

    # ─── Spectral Decomposition (from SEMPO) ───────────────────────────

    def adaptive_energy_mask(self, z):
        bs = z.shape[0]
        energy = torch.abs(z).pow(2).sum(dim=-1)
        flat_energy = energy.view(bs, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(bs, 1)
        normalized_energy = energy / (median_energy + 1e-6)
        energy_mask = ((normalized_energy > self.theta).float()
                       - self.theta).detach() + self.theta
        return energy_mask.unsqueeze(-1)

    def adaptive_frequency_mask(self, z, tau, mu):
        freq_num, bs, freq_len, n_vars = z.shape
        freq_indices = torch.arange(freq_len, device=z.device).unsqueeze(0).unsqueeze(2).expand(bs, -1, n_vars)
        tau = tau.view(freq_num, 1, 1, 1).expand(-1, bs, freq_len, n_vars)
        mu_exp = mu.view(freq_num, 1, freq_len, 1).expand(-1, bs, -1, n_vars)
        return torch.where(mu_exp == 1,
                           (freq_indices < tau).float(),
                           (freq_indices >= tau).float())

    def decomposed_frequency_learning(self, x):
        bs, seq_len, n_vars = x.shape
        z = torch.fft.rfft(x, dim=1, norm='ortho')
        energy_mask = self.adaptive_energy_mask(z)
        z_main = z * energy_mask
        z_res = z - z_main
        z_res = z_res.unsqueeze(0).expand(self.freq_num, -1, -1, -1)
        z_main = z_main.unsqueeze(0).expand(self.freq_num, -1, -1, -1)
        main_freq_mask = self.adaptive_frequency_mask(z_main, self.tau_main, self.mu_main)
        res_freq_mask = self.adaptive_frequency_mask(z_res, self.tau_res, self.mu_res)
        z = z_main * main_freq_mask + z_res * res_freq_mask
        return torch.fft.irfft(z, n=seq_len, dim=2, norm='ortho')

    # ─── Multi-view patching ──────────────────────────────────────────

    def _patch_view(self, x_view):
        """x_view: [B, T, C] → patches: [B*C, patch_num, D]"""
        return self.patch_embedding(x_view.permute(0, 2, 1))

    # ─── Forecast ──────────────────────────────────────────────────────

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, C = x_enc.shape[0], x_enc.shape[2]

        # ── Spectral decomposition + multi-view patching ──
        if self.freq_num > 0:
            x_freq = self.decomposed_frequency_learning(x_enc)  # [F, B, T, C]
        all_patches = []
        cls_preds = []

        # Raw time-domain view
        patches, _, cp = self._patch_view(x_enc)
        all_patches.append(patches)
        cls_preds.append(cp)

        # Frequency views
        for i in range(self.freq_num):
            patches, _, cp = self._patch_view(x_freq[i])
            all_patches.append(patches)
            cls_preds.append(cp)

        cls_pred = cls_preds[0]  # use raw view for classification loss

        # Cross-view aggregation: concat features → project
        views = torch.stack(all_patches, dim=0)  # [V, B*C, P, D]
        V, BxC, P, D = views.shape
        views = views.permute(1, 2, 0, 3)        # [B*C, P, V, D]
        views = views.reshape(BxC, P, V * D)
        enc_out = self.view_projection(views)     # [B*C, P, D]

        # ── Channel strategy → extra_token ──
        if self.channel == "CI":
            extra_token = self.enc_embedding(x_enc, None)
            extra_token = extra_token.view(-1, 1, self.d_model)
        elif self.channel == "CD":
            x_pool = x_enc.mean(dim=2)
            extra_token = self.enc_embedding(x_pool.unsqueeze(-1), None)
            extra_token = extra_token.mean(dim=1, keepdim=True)
            extra_token = extra_token.repeat_interleave(C, dim=0)
        elif self.channel == "CDP":
            x_pool = x_enc.mean(dim=2)
            channel_token = self.enc_embedding(x_pool.unsqueeze(-1), None)
            channel_token = channel_token.mean(dim=1, keepdim=True)
            channel_token = channel_token.repeat_interleave(C, dim=0)
            global_tokens = self.enc_embedding(x_enc, x_mark_enc)
            cal_tokens = global_tokens[:, C:, :].repeat_interleave(C, dim=0)
            extra_token = torch.cat([channel_token, cal_tokens], dim=1)
        elif self.channel == "CDA":
            extra_token = self.enc_embedding(x_enc, x_mark_enc)
            extra_token = extra_token.repeat_interleave(C, dim=0)
        elif self.channel == "CI+":
            global_tokens = self.enc_embedding(x_enc, x_mark_enc)
            var_tokens = global_tokens[:, :C, :].reshape(-1, 1, self.d_model)
            cal_tokens = global_tokens[:, C:, :].repeat_interleave(C, dim=0)
            extra_token = torch.cat([var_tokens, cal_tokens], dim=1)

        # ── MoE dynamic prompts ──
        prompts, moe_prompt_weights = self.prompt_moe(enc_out, extra_token)

        # ── MoE attention prefix ──
        if self.use_prefix and self.prefix_moe is not None:
            patch_pooled = enc_out.mean(dim=1)
            extra_pooled = extra_token.mean(dim=1)
            content = torch.cat([patch_pooled, extra_pooled], dim=-1)
            prefix_kv, moe_prefix_weights = self.prefix_moe(content)
        else:
            prefix_kv, moe_prefix_weights = None, None

        # ── Reconstruction (auxiliary) ──
        if self.use_reconstruct and self.reconstruct_head is not None:
            enc_recon = enc_out.clone()  # [B*C, P, D]
            enc_recon = enc_recon.view(B, C, D, -1)  # [B, C, D, patch_num]
            recon_out = self.reconstruct_head(enc_recon)  # [B, C, seq_len]
        else:
            recon_out = None

        # ── Concatenate enc_out with extra_token ──
        enc_out = torch.cat([enc_out, extra_token], dim=1)

        # ── Segment-wise prediction ──
        seg_outputs = []
        for i in range(self.num_segs):
            prompt = prompts[i]
            segment_input = torch.cat([prompt, enc_out], dim=1)
            segment_out, _ = self.encoder(segment_input, prefix_kv=prefix_kv)
            segment_out = segment_out[:, self.num_latent_token:
                                         self.num_latent_token + self.patch_num, :]
            segment_out = torch.reshape(segment_out, (B, C, self.d_model, self.patch_num))
            seg_outputs.append(self.heads[i](segment_out))

        dec_out = torch.cat(seg_outputs, dim=2).permute(0, 2, 1)

        # De-normalize
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return (dec_out, cls_pred, moe_prompt_weights,
                moe_prefix_weights, recon_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        (dec_out, cls_pred, moe_prompt_weights,
         moe_prefix_weights, recon_out) = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.training > 0:
            return (dec_out[:, -self.pred_len:, :], cls_pred,
                    moe_prompt_weights, moe_prefix_weights, recon_out)
        else:
            return dec_out[:, -self.pred_len:, :], cls_pred
