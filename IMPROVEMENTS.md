
# TimeMosaic-Fusion 改进说明

## 核心改进一览

| 组件 | TimeMosaic | TimeMosaic-Fusion |
|------|------------|------------------|
| 频谱分解 | ❌ | ✅ |
| 多视图聚合 | ❌ | ✅ |
| 提示生成 | Embedding | MoE动态生成 |
| 前缀K/V注入 | ❌ | ✅MoE生成 |
| 掩码重建 | ✅ | ✅ |
| 完整重建 | ❌ | ✅ |

---

## 详细改进点

### 1. 融合 SEMPO 的频谱分解（Spectral Decomposition）

#### 新增功能
- `decomposed_frequency_learning` 函数（L386-L397）：
  - 使用 RFFT 将信号分解到频域
  - 自适应能量掩码（`adaptive_energy_mask`，L366-L375）
  - 自适应频率掩码（`adaptive_frequency_mask`，L377-L384）
- 参数化的频域掩码：`tau_main`/`mu_main`/`tau_res`/`mu_res` 都是可学习参数
- 支持 4 个频域视图 + 原始时域视图

#### 目的
- 捕捉时间序列的频域周期性，与时域特征互补
- 可学习的掩码避免了固定频率阈值的局限性

---

### 2. 多视图聚合（Cross-view Aggregation）

#### 新增功能
- 对原始时域和 4 个频域视图分别做自适应分块
- 通过 `view_projection` 线性层将多个视图的特征融合（L288-L289、L437-L442）

#### 实现细节
```python
views = torch.stack(all_patches, dim=0)  # [V, B*C, P, D]
views = views.permute(1, 2, 0, 3)        # [B*C, P, V, D]
views = views.reshape(BxC, P, V * D)
enc_out = self.view_projection(views)     # [B*C, P, D]
```

#### 目的
- 轻量级融合，不显著增加参数量
- 保留各视图的互补性

---

### 3. MoE 动态提示（MoE Dynamic Prompt）

#### 新增功能
- 使用 `MoEPromptGenerator` 替代原 `nn.Embedding` 固定提示（L292-L300）
- 需要调参：`num_moe_experts`（默认 8）

#### 实现细节
- 基于 `enc_out` 和 `extra_token` 的混合特征进行专家路由
- 每个分段有独立的动态提示

#### 目的
- 替代固定 embedding，根据输入内容生成个性化提示
- 更灵活适应不同任务/数据集

---

### 4. MoE 前缀 K/V 注入（MoE Prefix for Attention K/V）

#### 新增功能
- `PrefixEncoder` 封装器（L64-L79）：支持向每层 Transformer 注入独立的前缀 K/V
- `MoEPrefixGenerator`（L164-L201）：通过 MoE 动态生成每层的前缀 K/V
- 需要调参：`num_moe_prefix_experts`（默认 4）、`prefix_len`（默认 4）

#### 实现细节
```python
if prefix_kv is not None:
    k = torch.cat([prefix_kv[0], x], dim=1)
    v = torch.cat([prefix_kv[1], x], dim=1)
```

#### 目的
- 首次在 TSF 中将 MoE 前缀直接注入到每一层的 K/V
- 前缀 K/V 与动态提示解耦，可独立调优

---

### 5. 新增独立的完整重建辅助任务

#### 背景
- TimeMosaic 已有**掩码重建**（通过 `mask_ratio` 或 `mask_ratio_patch` 配置）
- TimeMosaic-Fusion 在此基础上新增了**完整重建**（`reconstruct_head`，L348-L354）

#### TimeMosaic-Fusion 的完整重建实现
- 不需要 mask，直接重建整个原始输入信号
- 轻量级 MLP：`Linear(d_model, d_model*2) → ReLU → Linear(d_model*2, seq_len)`
- 需要配置：`use_reconstruct`（默认 True）

#### 目的
- 双重自监督目标：掩码重建 + 完整重建，增强特征鲁棒性
- 完整重建可以看作是一个额外的“正则化项”

---

### 6. EncoderLayer 增强

#### 修改点
- `EncoderLayer.forward` 新增 `prefix_kv` 参数（L39）
- 如果有前缀 K/V 就拼到 attention 的 K/V 上

---

### 7. 前向流程变化

#### TimeMosaic 流程
1. 归一化
2. 自适应分块（只时域）
3. 通道策略 → `extra_token`
4. 固定提示生成
5. Encoder 前向（无前缀注入）
6. 分段预测
7. 反归一化

#### TimeMosaic-Fusion 流程
1. 归一化
2. **频谱分解 → 多视图**
3. **每个视图分别自适应分块**
4. **多视图聚合**
5. 通道策略 → `extra_token`
6. **MoE 动态提示生成**
7. **MoE 前缀 K/V 生成**
8. **重建辅助任务前向**
9. Encoder 前向（带前缀注入）
10. 分段预测
11. 反归一化

---

## 实验效果总结

- 总任务数：36
- Fusion 优于 baseline：25（69%）
- Fusion 退化：11（31%）
- 最大提升：Exchange 320_720，MSE 下降 -24.64%
- 最大退化：ETTm2 320_96，MSE 上升 +1.70%

---

## 创新点提炼

1. **多视图自适应分块框架**：同时利用时域和可学习的频域视图
2. **双 MoE 模块**：动态提示 + 前缀 K/V 注入，解耦调优
3. **重建辅助任务**：增强特征鲁棒性

---

## 文件变更清单

- 新增/修改核心模型：`models/TimeMosaic_Fusion.py`
- 新增调参脚本：`scripts/TimeMosaic_Fusion/tune_moe.sh`
- 补充数据集脚本：`scripts/TimeMosaic_Fusion/Wind1.sh`、`Wind2.sh`、`Wind3.sh`、`Wind4.sh`、`ECL.sh`、`Traffic.sh`、`Solar.sh`、`PEMS.sh`
- 更新已有脚本：`scripts/TimeMosaic_Fusion/ETTm2.sh`（新增 320_* 任务）

---

## 使用说明

### 调参
```bash
# 运行所有数据集
bash scripts/TimeMosaic_Fusion/tune_moe.sh

# 只跑特定数据集
bash scripts/TimeMosaic_Fusion/tune_moe.sh ETTm1 ETTm2

# 干跑（打印命令不执行）
bash scripts/TimeMosaic_Fusion/tune_moe.sh --dry-run ETTm1
```

### 主要调参项
- `num_moe_experts`：MoE 提示专家数（推荐范围：2-8）
- `num_moe_prefix_experts`：MoE 前缀专家数（推荐范围：2-4）
- `prefix_len`：前缀长度（推荐：4）
- `use_reconstruct`：是否用重建辅助任务（推荐：True）
- `use_prefix`：是否用 MoE 前缀（推荐：True）

---

## 论文贡献点对应

| 论文贡献点 | 实现位置 |
|------------|----------|
| 多视图自适应分块框架 | L268-L289、L419-L442 |
| 双 MoE 动态适配机制 | L292-L315、L470-L480 |
| 多个数据集验证效果 | `TimeMosaic_Fusion_best.txt` + `原TimeMosaic.txt` |

