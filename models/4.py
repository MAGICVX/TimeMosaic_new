# 三层层次化 Mixture-of-Experts 架构
from layers.Transformer_EncDec import Encoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.revin import RevIN
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_latent_token=0):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.num_latent_token = num_latent_token

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        q = self.mask_last_tokens(x)
        new_x, attn = self.attention(
            q, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
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


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class AdaptivePatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len_list, mode='fixed', dropout=0.0, seq_len=96, in_channels=1, training=True):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.mode = mode
        self.max_patch_len = max(patch_len_list)
        self.min_patch_len = min(patch_len_list)
        self.region_num = seq_len // self.max_patch_len
        self.d_model = d_model
        self.in_channels = in_channels
        self.training = training

        self.register_buffer('target_ratio', torch.ones(len(patch_len_list)) / len(patch_len_list))

        self.region_cls = nn.Sequential(
            nn.Linear(self.max_patch_len, 64),
            nn.ReLU(),
            nn.Linear(64, len(patch_len_list))
        )

        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model, bias=False) for patch_len in patch_len_list
        ])

        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, L = x.shape
        assert L == self.region_num * self.max_patch_len, \
            f"Expected seq_len={self.region_num * self.max_patch_len}, but got {L}"

        x = x.reshape(B * C, self.region_num, self.max_patch_len)

        all_patches = []
        cls_pred_list = []
        cls_soft_list = []

        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]

            cls_logits = self.region_cls(region)

            if self.training:
                cls_soft = F.gumbel_softmax(cls_logits, tau=0.5, hard=True, dim=-1)
            else:
                cls_pred = torch.argmax(cls_logits, dim=-1)
                cls_soft = F.one_hot(cls_pred, num_classes=len(self.patch_len_list)).float()

            cls_soft_list.append(cls_soft)

            cls_pred = cls_soft.argmax(dim=-1)
            cls_pred_list.append(cls_pred)

            patch_emb_list = []
            for idx, patch_len in enumerate(self.patch_len_list):
                patches = region.unfold(-1, patch_len, patch_len)
                if self.mode == 'fixed':
                    target_patch_num = self.max_patch_len // self.min_patch_len
                    repeat = target_patch_num - patches.size(1)
                    if repeat > 0:
                        patches = patches.repeat_interleave(repeat + 1, dim=1)[:, :target_patch_num, :]
                patches_emb = self.embeddings[idx](patches)
                patch_emb_list.append(patches_emb)

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


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
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


# ============================================================
# Level 1: Segment-level Mixture-of-Experts (MoE)
# ============================================================

class TrendExpert(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DilatedAttentionLayer(d_model, n_heads, d_ff, dropout, dilation=2 ** i)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x, attn_mask)
        return self.norm(x)


class PeriodicExpert(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers=2, dropout=0.1):
        super().__init__()
        self.freq_proj = nn.Linear(d_model, d_model)
        self.attn_layers = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(FullAttention(False, 1, attention_dropout=dropout,
                                             output_attention=False), d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        freq_feat = self._freq_modulation(x)
        x = x + freq_feat
        for layer in self.attn_layers:
            x, _ = layer(x, attn_mask)
        return self.norm(x)

    def _freq_modulation(self, x):
        B, L, D = x.shape
        x_centered = x - x.mean(dim=1, keepdim=True)
        x_var = torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-8
        x_normed = x_centered / torch.sqrt(x_var)
        x_fft = torch.fft.rfft(x_normed.float(), dim=-1)
        freq_mag = torch.abs(x_fft)
        if freq_mag.size(-1) < D:
            freq_feat = F.pad(freq_mag, (0, D - freq_mag.size(-1)))
        else:
            freq_feat = freq_mag[..., :D]
        return self.freq_proj(freq_feat)


class AbruptExpert(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DilatedAttentionLayer(d_model, n_heads, d_ff, dropout, dilation=d+1)
            for d in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x, _ = layer(x)
        return self.norm_out(x)


class DilatedAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.attention = AttentionLayer(
            FullAttention(False, 1, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, _ = self.attention(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(new_x))
        residual = x
        x = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        x = self.conv2(x).transpose(1, 2)
        return self.norm2(residual + self.dropout(x)), None


class LocalWindowAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attention = AttentionLayer(
            FullAttention(False, 1, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, L, D = x.shape
        w = self.window_size
        padded = F.pad(x, (0, 0, w // 2, w // 2))
        windows = padded.unfold(1, w, 1)  # [B, L, w, D]
        B_win, L_win, W, D_win = windows.shape
        windows_flat = windows.reshape(B_win * L_win, W, D_win)
        out_win, _ = self.attention(windows_flat, windows_flat, windows_flat, None)
        new_x = out_win.reshape(B_win, L_win, W, D_win).mean(dim=2)
        x = self.norm1(x + self.dropout(new_x))
        residual = x
        x = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        x = self.conv2(x).transpose(1, 2)
        return self.norm2(residual + self.dropout(x)), None


class SegmentGatingNetwork(nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, num_experts)
        )

    def forward(self, x):
        gate_logits = self.gate(x)  # [B*C, num_experts]
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        zeros = torch.zeros_like(gate_logits)
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)
        return gates, gate_logits


class LoadBalanceLoss(nn.Module):
    def __init__(self, num_experts, loss_coef=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.loss_coef = loss_coef

    def forward(self, gate_logits):
        gating_decision = F.softmax(gate_logits, dim=-1).mean(dim=0)
        return self.loss_coef * (self.num_experts * gating_decision.var() + (1 / self.num_experts))


# ============================================================
# Level 2: Global Context Aggregation
# ============================================================

class GlobalContextModule(nn.Module):
    def __init__(self, d_model, num_segs, n_heads=8, dropout=0.1):
        super().__init__()
        self.num_segs = num_segs

        self.seg_pooling = nn.AdaptiveAvgPool1d(1)
        self.seg_proj = nn.Linear(d_model, d_model)

        self.global_transformer = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, 1, attention_dropout=dropout,
                                             output_attention=False), d_model, n_heads),
                d_model, d_model * 2, dropout=dropout, activation="gelu"
            ) for _ in range(2)
        ], norm_layer=nn.LayerNorm(d_model))

        self.cross_seg_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

    def forward(self, seg_representations):
        """
        seg_representations: List of [B*C, d_model] per segment
        Returns: global_context [B*C, d_model], all_seg_features [num_segs, B*C, d_model]
        """
        seg_features = []
        for i, seg_repr in enumerate(seg_representations):
            projected = self.seg_proj(seg_repr)
            seg_features.append(projected)

        seg_stack = torch.stack(seg_features, dim=0)  # [num_segs, B*C, d_model]

        global_ctx, _ = self.global_transformer(seg_stack)
        global_summary = global_ctx.mean(dim=0)  # [B*C, d_model]

        refined_segs = []
        for i in range(self.num_segs):
            refined, _ = self.cross_seg_attn(
                query=seg_features[i].unsqueeze(1),
                key=global_summary.unsqueeze(1),
                value=global_summary.unsqueeze(1)
            )
            refined_segs.append(refined.squeeze(1))

        return global_summary, refined_segs, seg_stack


class GlobalPredictionHead(nn.Module):
    def __init__(self, d_model, pred_len, enc_in, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len)
        )

    def forward(self, global_context):
        B_C, D = global_context.shape
        out = self.mlp(global_context)
        B_total = B_C // self.enc_in
        out = out.view(B_total, self.enc_in, self.pred_len)
        return out


# ============================================================
# Cross-Level Connection: Top-Down Refinement
# ============================================================

class TopDownAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, local_repr, global_context):
        """
        local_repr: [B*C, L_local, d_model] per segment
        global_context: [B*C, d_model]
        Returns: refined local representation
        """
        B_C, L, D = local_repr.shape
        g = global_context.unsqueeze(1)  # [B*C, 1, D]

        q = self.query_proj(local_repr).view(B_C, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(g).view(B_C, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(g).view(B_C, 1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B_C, L, D)
        out = self.out_proj(out)

        return self.norm(local_repr + out)


class MultiScaleFusionGate(nn.Module):
    def __init__(self, d_model, num_segs, pred_len, enc_in):
        super().__init__()
        self.num_segs = num_segs
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_proj = nn.ModuleList([
            nn.Linear(enc_in, enc_in) for _ in range(num_segs)
        ])
        gate_in_dim = enc_in * (num_segs + 1)
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_segs + 1),
            nn.Softmax(dim=-1)
        )

    def forward(self, local_outputs, global_output):
        B = local_outputs[0].size(0)
        local_upsampled = []
        for i, lo in enumerate(local_outputs):
            lo_t = lo.permute(0, 2, 1)
            if lo_t.size(1) != self.pred_len:
                lo_t = F.interpolate(lo_t.transpose(1, 2), size=self.pred_len, mode='linear', align_corners=False).transpose(1, 2)
            proj_lo = self.seg_proj[i](lo_t)
            local_upsampled.append(proj_lo)

        global_t = global_output.permute(0, 2, 1)

        stacked = torch.stack(local_upsampled + [global_t], dim=1)
        gate_input = stacked.mean(dim=2).flatten(start_dim=1)
        gates = self.gate_net(gate_input)

        fused = (gates.unsqueeze(-1).unsqueeze(-1) * stacked).sum(dim=1)
        return fused


# ============================================================
# Hierarchical Mosaic Transformer Model
# ============================================================

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.training_flag = configs.is_training

        self.patch_len_list = eval(configs.patch_len_list)

        seg_len_map = {96: configs.pre96, 192: configs.pre192, 336: configs.pre336, 720: configs.pre720}
        self.seg_len = seg_len_map.get(configs.pred_len, configs.pre12)
        self.num_segs = self.pred_len // self.seg_len
        self.channel = configs.channel

        # ===== Level 0: Multi-Granularity Patch Tokenization =====
        self.patch_embedding = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            patch_len_list=self.patch_len_list,
            mode='fixed',
            dropout=configs.dropout,
            seq_len=configs.seq_len,
            in_channels=configs.enc_in,
            training=configs.is_training
        )

        self.num_latent_token = configs.num_latent_token
        self.prompt_embeddings = nn.Embedding(self.num_latent_token * self.num_segs, self.d_model)
        nn.init.xavier_uniform_(self.prompt_embeddings.weight)

        # Shared encoder backbone (used before MoE routing)
        self.shared_encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                num_latent_token=configs.num_latent_token,
            ) for l in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)))

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        # ===== Level 1: Segment-level MoE =====
        self.num_moe_experts = 3
        self.moe_top_k = 2

        self.segment_experts = nn.ModuleDict({
            'trend': TrendExpert(configs.d_model, configs.n_heads, configs.d_ff,
                                 num_layers=1, dropout=configs.dropout),
            'periodic': PeriodicExpert(configs.d_model, configs.n_heads, configs.d_ff,
                                       num_layers=1, dropout=configs.dropout),
            'abrupt': AbruptExpert(configs.d_model, configs.n_heads, configs.d_ff,
                                   num_layers=1, dropout=configs.dropout),
        })
        self.expert_names = ['trend', 'periodic', 'abrupt']

        self.seg_gating_network = SegmentGatingNetwork(
            configs.d_model, self.num_moe_experts, top_k=self.moe_top_k
        )
        self.load_balance_loss = LoadBalanceLoss(self.num_moe_experts)

        # ===== Prediction Heads per segment =====
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)

        self.heads = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf, self.seg_len, head_dropout=configs.dropout)
            for _ in range(self.num_segs)
        ])

        self.moe_heads = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf, self.seg_len, head_dropout=configs.dropout)
            for _ in range(self.num_segs)
        ])

        self.enhance_gate = nn.Parameter(torch.tensor(-5.0))
        self.max_alpha = 0.1

        # ===== Normalization =====
        self.revin = False
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # ===== Mask =====
        self.mask_ratio = getattr(configs, "mask_ratio", 0)
        self.mask_ratio_patch = getattr(configs, "mask_ratio_patch", 0)
        self.mask_reconstruct_head = None

        if self.mask_ratio > 0:
            self.mask_ratio_patch = 0
            self.mask_reconstruct_head = FlattenHead(
                configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout
            )
        elif self.mask_ratio_patch > 0:
            self.mask_ratio = 0

        self.latest_gate_logits = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # === Normalization ===
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, C = x_enc.shape[0], x_enc.shape[2]

        if self.mask_ratio > 0 and self.training_flag:
            x_enc[mask] = 0.0

        # === Channel Embedding ===
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
            cal_tokens = global_tokens[:, C:, :]
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0)
            extra_token = torch.cat([channel_token, cal_tokens], dim=1)
        elif self.channel == "CDA":
            extra_token = self.enc_embedding(x_enc, x_mark_enc)
            extra_token = extra_token.repeat_interleave(C, dim=0)
        elif self.channel == "CI+":
            global_tokens = self.enc_embedding(x_enc, x_mark_enc)
            var_tokens = global_tokens[:, :C, :]
            cal_tokens = global_tokens[:, C:, :]
            var_tokens = var_tokens.reshape(-1, 1, self.d_model)
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0)
            extra_token = torch.cat([var_tokens, cal_tokens], dim=1)

        # === Level 0: Patch Embedding ===
        x_enc_permuted = x_enc.permute(0, 2, 1)
        enc_out, n_vars, cls_pred = self.patch_embedding(x_enc_permuted)

        # Mask handling
        dec_mask = None
        if self.mask_ratio > 0 and self.training_flag:
            enc_mask = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
            enc_mask = enc_mask.permute(0, 1, 3, 2)
            dec_mask = self.mask_reconstruct_head(enc_mask)
            dec_mask = dec_mask.permute(0, 2, 1)
        elif self.mask_ratio_patch > 0 and self.training_flag:
            enc_out[mask] = 0.0

        enc_out = torch.cat([enc_out, extra_token], dim=1)

        # === Level 0: Shared Encoding (SAME AS ORIGINAL TimeMosaic) ===
        seg_raw_outputs = []
        for i in range(self.num_segs):
            prompt = self.prompt_embeddings.weight[i * self.num_latent_token:(i + 1) * self.num_latent_token]
            prompt = prompt.unsqueeze(0).expand(B * C, -1, -1)
            segment_input = torch.cat([prompt, enc_out], dim=1)
            segment_out, _ = self.shared_encoder(segment_input)
            segment_out = segment_out[:, self.num_latent_token:self.num_latent_token + self.patch_num, :]
            segment_out_reshaped = torch.reshape(segment_out, (B, C, self.d_model, self.patch_num))
            seg_raw_outputs.append(segment_out_reshaped)

        # === BASELINE PREDICTION (identical to original TimeMosaic) ===
        baseline_seg_outputs = []
        for i in range(self.num_segs):
            seg_out_i = self.heads[i](seg_raw_outputs[i])
            baseline_seg_outputs.append(seg_out_i)
        baseline_pred = torch.cat(baseline_seg_outputs, dim=2).permute(0, 2, 1)

        # === MoE Enhancement (lightweight residual) ===
        moe_loss = 0.0
        all_gate_logits = []
        seg_representations = []

        for i in range(self.num_segs):
            B_s, C_s, D_s, P_s = seg_raw_outputs[i].shape
            seg_seq = seg_raw_outputs[i].permute(0, 1, 3, 2).reshape(B_s * C_s, P_s, D_s)
            seg_pooled = seg_seq.mean(dim=1)

            gates, gate_logits = self.seg_gating_network(seg_pooled)
            all_gate_logits.append(gate_logits)

            expert_outputs = {}
            for name in self.expert_names:
                expert_outputs[name] = self.segment_experts[name](seg_seq)

            combined = torch.zeros_like(seg_seq)
            for j, name in enumerate(self.expert_names):
                weight = gates[:, j:j+1].unsqueeze(1)
                combined = combined + weight * expert_outputs[name]

            combined_reshaped = combined.reshape(B_s, C_s, P_s, D_s).permute(0, 1, 3, 2)
            moe_seg_out = self.moe_heads[i](combined_reshaped)
            seg_representations.append(moe_seg_out)

        self.latest_gate_logits = all_gate_logits
        if self.training_flag:
            for gl in all_gate_logits:
                moe_loss = moe_loss + self.load_balance_loss(gl)
            moe_loss = moe_loss / len(all_gate_logits)

        # === MoE-enhanced prediction ===
        moe_pred = torch.cat(seg_representations, dim=2).permute(0, 2, 1)

        if moe_pred.size(1) != self.pred_len:
            if moe_pred.size(1) < self.pred_len:
                moe_pred = F.pad(moe_pred, (0, 0, 0, self.pred_len - moe_pred.size(1)))
            else:
                moe_pred = moe_pred[:, :self.pred_len, :]

        # === CONSERVATIVE FUSION: baseline + small MoE correction ===
        effective_alpha = self.max_alpha * torch.sigmoid(self.enhance_gate)
        dec_out = baseline_pred + effective_alpha * (moe_pred - baseline_pred)

        # === De-normalization ===
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out, cls_pred, dec_mask, moe_loss

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, cls_pred, dec_mask, moe_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.training_flag > 0:
            return dec_out[:, -self.pred_len:, :], cls_pred, dec_mask, moe_loss
        else:
            return dec_out[:, -self.pred_len:, :], cls_pred.permute(0, 2, 1)
