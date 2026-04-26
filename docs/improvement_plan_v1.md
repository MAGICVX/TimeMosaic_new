# TimeMosaic 模型改进方案

## 方案一：递归隐藏状态传递（Recursive Hidden State Passing）


## 方向 1：动态 Prompt 学习机制
问题：当前 prompt embeddings 是静态学习的 (nn.Embedding)，无法适应不同输入模式

创新点：

Input-aware Prompt Generation: 根据输入序列的动态特征生成 prompt

### 方向 2：多尺度时序异质性建模
问题 ：当前的 patch 分类只考虑单一时间尺度，忽略了多尺度特征

创新点 ：

- Hierarchical Patch Classification : 引入多粒度决策
# 第一层：粗粒度 (日/周级别)
coarse_cls = self.coarse_classifier(region)
# 第二层：细粒度 (小时级别)
fine_cls = self.fine_classifier(region, coarse_cls)
- Cross-scale Attention : 不同 patch 长度之间的信息交互
# 让不同粒度的 patch 互相通信
multi_scale_attn = CrossScaleAttention(
    scales=[16, 32, 64],
    d_model=d_model
)

### 方向 3：不确定性量化预测
问题 ：当前模型只输出点预测，没有不确定性估计

创新点 ：

- Probabilistic Forecasting Head : 输出分布参数而非点值


### 方向 4：频域自适应 Patching
问题 ：当前 patch 选择仅基于时域特征，忽略了频域信息

创新点 ：

- Frequency-guided Patch Selection :
def frequency_analysis(x):
    # FFT 变换
    x_freq = torch.fft.rfft(x, dim=-1)
    dominant_freq = torch.argmax(torch.abs(x_freq), dim=-1)
    # 根据主频选择 patch 长度
    patch_len = freq_to_patch_len(dominant_freq)
- Dual-domain Encoder : 同时处理时域和频域特征
class DualDomainEncoder(nn.Module):
    def forward(self, x_time, x_freq):
        time_feat = self.time_encoder(x_time)
        freq_feat = self.freq_encoder(x_freq)
        # 跨域融合
        fused = self.cross_domain_attention(time_feat, freq_feat)


第1步 → 卷积增强 Patch Embedding  (几乎无副作用)
    ↓
第2步 → Enhanced FlattenHead       (解码器改进)
    ↓
第3步 → Dynamic Input-Conditioned Prompt (核心创新)
    ↓
第4步 → Cross-Region Interaction   (信息流增强)
    ↓
第5步 → Relative Position Bias    (可选，看提升幅度)