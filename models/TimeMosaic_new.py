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

    def forward(self, x):  # x: [B*C, num_patch, d_model]
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
        # self.position_embedding = nn.Parameter(torch.ones(1, 1000, d_model))  # [1, max_patches, d_model]
        # nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        assert L == self.region_num * self.max_patch_len, \
            f"Expected seq_len={self.region_num * self.max_patch_len}, but got {L}"

        x = x.reshape(B * C, self.region_num, self.max_patch_len)  # [B*C, R, max_patch_len]

        all_patches = []
        cls_pred_list = []
        cls_soft_list = []

        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]  # [B*C, max_patch_len]

            cls_logits = self.region_cls(region)  # [B*C, num_classes]

            if self.training:
                cls_soft = F.gumbel_softmax(cls_logits, tau=0.5, hard=True, dim=-1)  # [B*C, num_classes]
            else:
                cls_pred = torch.argmax(cls_logits, dim=-1)
                cls_soft = F.one_hot(cls_pred, num_classes=len(self.patch_len_list)).float()

            cls_soft_list.append(cls_soft)

            cls_pred = cls_soft.argmax(dim=-1)
            cls_pred_list.append(cls_pred)

            patch_emb_list = []
            for idx, patch_len in enumerate(self.patch_len_list):
                patches = region.unfold(-1, patch_len, patch_len)  # [B*C, num_patch, patch_len]
                if self.mode == 'fixed':
                    target_patch_num = self.max_patch_len // self.min_patch_len
                    repeat = target_patch_num - patches.size(1)
                    if repeat > 0:
                        patches = patches.repeat_interleave(repeat + 1, dim=1)[:, :target_patch_num, :]
                patches_emb = self.embeddings[idx](patches)  # [B*C, num_patch, d_model]
                patch_emb_list.append(patches_emb)

            # [num_classes, B*C, num_patch, d_model]
            patch_emb_stack = torch.stack(patch_emb_list, dim=0)

            # cls_soft: [B*C, num_classes] → [num_classes, B*C, 1, 1]
            cls_soft_trans = cls_soft.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

            region_patches_sorted = (patch_emb_stack * cls_soft_trans).sum(dim=0)  # [B*C, num_patch, d_model]

            all_patches.append(region_patches_sorted)

        x_patch = torch.cat(all_patches, dim=1)  # [B*C, total_num_patch, d_model]
        x_patch += self.position_embedding(x_patch)
        # x_patch = x_patch + self.position_embedding[:, :x_patch.size(1), :]
        x_patch = self.dropout(x_patch)

        all_cls_pred = torch.cat(cls_pred_list, dim=0)  # [B*C*R]
        self.latest_cls_soft = torch.cat(cls_soft_list, dim=0)

        return x_patch, C, all_cls_pred



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# ── Auxiliary Loss Functions ──────────────────────────────────────────

def compute_boundary_loss(dec_out_raw, seg_len):
    """
    Penalize discontinuities at segment boundaries.
    Segments are decoded independently; this forces adjacent boundary
    points to be similar, providing implicit cross-segment regularization.
    dec_out_raw: [B, C, pred_len] in normalized space.
    """
    num_segs = dec_out_raw.shape[-1] // seg_len
    if num_segs <= 1:
        return torch.tensor(0.0, device=dec_out_raw.device)
    loss = 0.0
    for i in range(num_segs - 1):
        end_i = dec_out_raw[:, :, (i + 1) * seg_len - 1]
        start_next = dec_out_raw[:, :, (i + 1) * seg_len]
        loss = loss + F.l1_loss(end_i, start_next)
    return loss / (num_segs - 1)


def compute_gradient_loss(pred_raw, y_norm):
    """
    First-order + second-order difference loss.
    Directly supervises trend direction and curvature to prevent
    long-horizon drift.
    pred_raw, y_norm: [B, C, pred_len].
    """
    pred_d1 = pred_raw[:, :, 1:] - pred_raw[:, :, :-1]
    true_d1 = y_norm[:, :, 1:] - y_norm[:, :, :-1]
    loss_d1 = F.l1_loss(pred_d1, true_d1)

    pred_d2 = pred_d1[:, :, 1:] - pred_d1[:, :, :-1]
    true_d2 = true_d1[:, :, 1:] - true_d1[:, :, :-1]
    loss_d2 = F.l1_loss(pred_d2, true_d2)

    return loss_d1 + 0.5 * loss_d2


def compute_spectral_loss(pred_raw, y_norm):
    """
    FFT amplitude-spectrum matching loss.
    Preserves periodic structure even when point-wise errors accumulate.
    pred_raw, y_norm: [B, C, pred_len].
    """
    pred_fft = torch.fft.rfft(pred_raw.float(), dim=-1)
    true_fft = torch.fft.rfft(y_norm.float(), dim=-1)
    return F.l1_loss(torch.abs(pred_fft), torch.abs(true_fft))


def compute_multiscale_loss(seg_outputs, y_norm, seg_len):
    """
    Per-segment supervision: each segment's output is independently
    compared to its target slice. Prevents intermediate horizon drift.
    seg_outputs: list of [B, C, seg_len];  y_norm: [B, C, pred_len].
    """
    loss = 0.0
    for i, seg_out in enumerate(seg_outputs):
        target = y_norm[:, :, i * seg_len:(i + 1) * seg_len]
        loss = loss + F.l1_loss(seg_out, target)
    return loss / len(seg_outputs)


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.training = configs.is_training

        self.patch_len_list = eval(configs.patch_len_list)

        seg_len_map = {96: configs.pre96, 192: configs.pre192, 336: configs.pre336, 720: configs.pre720}

        self.seg_len = seg_len_map.get(configs.pred_len, configs.pre12)

        self.num_segs = self.pred_len // self.seg_len
        self.channel = configs.channel

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

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                # open mask seq_len
                # num_latent_token=configs.num_latent_token,
            ) for l in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2)))

        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        
        self.heads = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf, self.seg_len, head_dropout=configs.dropout)
            for _ in range(self.num_segs)
        ])
        
        self.revin = False
        self.revin_layer = RevIN(configs.enc_in,affine=True,subtract_last=False)
        
        # mask
        self.mask_ratio = getattr(configs, "mask_ratio", 0)
        self.mask_ratio_patch = getattr(configs, "mask_ratio_patch", 0)
        self.mask_reconstruct_head = None

        if self.mask_ratio > 0:
            # 优先使用 token masking
            self.mask_ratio_patch = 0  # 明确互斥逻辑
            self.mask_reconstruct_head = FlattenHead(
                configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout
            )
        elif self.mask_ratio_patch > 0:
            # patch masking，只需要设定比例
            self.mask_ratio = 0



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask, y_target=None):
        # Normalization from Non-stationary Transformer
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        
        seg_outputs = []
        B, C = x_enc.shape[0], x_enc.shape[2]
        
        if self.mask_ratio > 0 and self.training:
            x_enc[mask] = 0.0
                
        # channel independent
        if self.channel == "CI":
            # simple CI
            extra_token = self.enc_embedding(x_enc, None)
            extra_token = extra_token.view(-1, 1, self.d_model)
        elif self.channel == "CD":
            # simple channel-wise 
            x_pool = x_enc.mean(dim=2)  # [B, T]
            extra_token = self.enc_embedding(x_pool.unsqueeze(-1), None)  # [B, T, d]
            extra_token = extra_token.mean(dim=1, keepdim=True)  # [B, 1, d]
            extra_token = extra_token.repeat_interleave(C, dim=0)  # [B*C, 1, d]
        elif self.channel == "CDP":
            # simple channel-wise 
            x_pool = x_enc.mean(dim=2)  # [B, T]
            channel_token = self.enc_embedding(x_pool.unsqueeze(-1), None)  # [B, T, d]
            channel_token = channel_token.mean(dim=1, keepdim=True)  # [B, 1, d]
            channel_token = channel_token.repeat_interleave(C, dim=0)  # [B*C, 1, d]

            global_tokens = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            cal_tokens = global_tokens[:, C:, :]
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0) 

            extra_token = torch.cat([channel_token, cal_tokens], dim=1)
        elif self.channel == "CDA":
            extra_token = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            extra_token = extra_token.repeat_interleave(C, dim=0)  # [B*C, C+K, D]
        elif self.channel == "CI+":
            global_tokens = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            var_tokens = global_tokens[:, :C, :]
            cal_tokens = global_tokens[:, C:, :]

            var_tokens = var_tokens.reshape(-1, 1, self.d_model)          # [B*C, 1, D]
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0)          # [B*C, K, D]

            extra_token = torch.cat([var_tokens, cal_tokens], dim=1)

        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars, cls_pred = self.patch_embedding(x_enc)
        
        # mask
        if self.mask_ratio > 0 and self.training:
            enc_mask = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
            # z: [bs x nvars x d_model x patch_num]
            enc_mask = enc_mask.permute(0, 1, 3, 2)
            # Decoder
            dec_mask = self.mask_reconstruct_head(enc_mask)
            dec_mask = dec_mask.permute(0, 2, 1)
        elif self.mask_ratio_patch > 0 and self.training:
            enc_out[mask] = 0.0
            dec_mask = None
        else:
            dec_mask = None
        
        # enc_out shape: torch.Size([B * C, patch_num, d_model])
        enc_out = torch.cat([enc_out, extra_token], dim=1)
        

        for i in range(self.num_segs):
            # Prompt embedding for segment i
            prompt = self.prompt_embeddings.weight[i * self.num_latent_token : (i + 1) * self.num_latent_token]
            prompt = prompt.unsqueeze(0).expand(B * C, -1, -1)  # [B*C, num_prompt, d_model]

            # Concatenate prompt, patch embeddings, global channel
            segment_input = torch.cat([prompt, enc_out], dim=1)  # [B*C, prompt+patch+1, d_model]

            # Encoder forward
            segment_out, _ = self.encoder(segment_input)
            segment_out = segment_out[:, self.num_latent_token:self.num_latent_token + self.patch_num, :]  # remove prompt

            # Reshape: [B*C, patch_num, d_model] → [B, C, d_model, patch_num]
            segment_out = torch.reshape(segment_out, (B, C, self.d_model, self.patch_num))

            # Segment head: [B, C, seg_len]
            seg_out = self.heads[i](segment_out)
            seg_outputs.append(seg_out)

        # Concatenate all segment outputs: [B, C, pred_len] (normalized space)
        dec_out_raw = torch.cat(seg_outputs, dim=2)  # [B, C, pred_len]
        dec_out = dec_out_raw.permute(0, 2, 1)       # → [B, pred_len, C]

        # De-normalize
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        # ── Compute auxiliary losses (training only) ──
        aux_losses = {}
        if self.training and y_target is not None and not self.revin:
            y_norm = (y_target[:, -self.pred_len:, :] - means) / (stdev + 1e-5)
            y_norm = y_norm.permute(0, 2, 1)  # [B, C, pred_len]

            aux_losses['boundary']   = compute_boundary_loss(dec_out_raw, self.seg_len)
            aux_losses['gradient']   = compute_gradient_loss(dec_out_raw, y_norm)
            aux_losses['spectral']   = compute_spectral_loss(dec_out_raw, y_norm)
            aux_losses['multiscale'] = compute_multiscale_loss(seg_outputs, y_norm, self.seg_len)

        return dec_out, cls_pred, dec_mask, aux_losses

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, y_target=None):
        dec_out, cls_pred, dec_mask, aux_losses = self.forecast(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask, y_target
        )
        if self.training > 0:
            return dec_out[:, -self.pred_len:, :], cls_pred, dec_mask, aux_losses
        else:
            return dec_out[:, -self.pred_len:, :], cls_pred