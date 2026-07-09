# ONNX 混合精度量化

ACT 模型的 INT8 + FP16 混合精度量化方案. 基于 666 帧全量评估, 纯 CPU EP
(AVX-VNNI 加速 INT8 GEMM, 无需 GPU).

## 评估方法

### 数据集

- 666 帧全量评估 (`output/dataset/videos/observation.images.fpv/chunk-000/*.jpg`)
- 参考: `output/infer_results_onnx.json` (FP32 ONNX 的完整推理结果)
- 指标: turn match (LEFT/RIGHT/STRAIGHT 方向一致率), max_diff (速度最大绝对误差)
- 推理设备: CPU EP (AVX-VNNI), 纯 CPU 量化路径, 嵌入式部署的真实场景

## 模型结构

ACT 模型 = ResNet18 视觉编码器 + Transformer 编码器/解码器, ONNX 导出后 677
个节点. `quant_pre_process` 会把 `MatMul + Add(bias)` 融合成 `Gemm`, 所以
FFN/QKV 投影在预处理后都是 Gemm.

| 类别 | 算子 | 数量 | 说明 |
|------|------|------|------|
| conv | Conv | 21 | ResNet18 卷积 |
| qkv | Gemm (MatMulAddFusion) | 32 | 注意力 Q/K/V 投影 |
| attn_out_proj | Gemm | 11 | 注意力输出投影 |
| ffn1 | Gemm | 8 | FFN 升维 512->3200 |
| ffn2 | Gemm | 8 | FFN 降维 3200->512 |
| attn_core | MatMul | 22 | Q@K^T + attn@V (softmax 附近) |
| tiny | Gemm/MatMul | 3 | action_head/state_encoder/latent_proj |

## 敏感度分析

### 方法

对每个类别单独做 INT8 静态量化 (其他全部 FP32), 跑 666 帧评估. 这能隔离每类
算子对 INT8 的固有敏感度. 校准用 MinMax + uniform 100 帧.

脚本: `scripts/sensitivity_analysis.py`

```bash
.venv/bin/python scripts/sensitivity_analysis.py
```

### 结果 (666 帧, CPU EP)

| 类别 | 节点数 | turn match | drop | max_diff | 结论 |
|------|--------|-----------|------|----------|------|
| qkv-only | 32 | 99.10% | +0.90% | 0.000462 | 可 INT8 |
| conv-only | 21 | 98.05% | +1.95% | 0.002235 | 可 INT8 |
| attn-out-only | 11 | 97.75% | +2.25% | 0.001958 | 边界, 保守 FP16 |
| ffn1-only | 8 | 94.29% | +5.71% | 0.003872 | 必须 FP16 |
| ffn2-only | 8 | 84.53% | +15.47% | 0.002753 | 必须 FP16 |

### 关键发现

1. **FFN (尤其 ffn2) 最敏感**: drop 5.71% / 15.47%. ffn2 (3200->512) 是信息瓶颈,
   INT8 量化误差经残差连接放大到整个 hidden_dim. ffn1 输出经 GELU, 在小值域精度差.
2. **qkv 最不敏感**: drop 0.90%. 线性变换, 权重分布均匀, INT8 损失极小.
3. **conv 中等**: drop 1.95%. ResNet18 Conv 有成熟 INT8 经验, per-channel 足够精确,
   但不如 qkv.
4. **attn_out 边界**: drop 2.25%. 介于 INT8/FP16 之间, 保守起见放 FP16.

## 逐层敏感度分析

### 方法

对按类别分析中"中间地带"的类别 (conv drop 1.95%, attn_out drop 2.25%) 做
leave-one-in: 每个层单独 INT8 (其余 FP32), 找出类内的坏分子. 明确结论的类别
(qkv 全安全, ffn 全敏感) 不测, 省时间.

脚本: `scripts/per_layer_sensitivity.py`

```bash
.venv/bin/python scripts/per_layer_sensitivity.py                   # 默认 conv + attn_out
.venv/bin/python scripts/per_layer_sensitivity.py --categories conv # 指定类别
```

### 结果 (666 帧, CPU EP)

| 类别 | 数量 | SAFE (<0.5%) | WARN (0.5~2%) | BAD (≥2%) | 中位 drop | 最大 drop |
|------|------|--------------|---------------|-----------|-----------|-----------|
| conv | 21 | 17 | 4 | 0 | +0.15% | +1.05% |
| attn_out_proj | 11 | 5 | 4 | 2 | +0.75% | +2.70% |

### 坏分子 (单层 leave-one-in drop ≥ 1.0%)

| 层 | 类别 | drop | max_diff |
|----|------|------|----------|
| `/decoder/layers.3/self_attn/Gemm` | attn_out | +2.70% | 0.001157 |
| `/decoder/layers.2/multihead_attn/Gemm` | attn_out | +2.55% | 0.000809 |
| `/decoder/layers.3/multihead_attn/Gemm` | attn_out | +1.95% | 0.001076 |
| `/decoder/layers.2/self_attn/Gemm` | attn_out | +1.20% | 0.000621 |
| `/vision_encoder/backbone/backbone.0/Conv` | conv | +1.05% | 0.001669 |
| `/vision_encoder/backbone/backbone.4/backbone.4.0/conv2/Conv` | conv | +1.05% | 0.001537 |

### 解读

1. **conv 类内分布均匀**: 17/21 是 SAFE, 无 BAD. 整类 drop 1.95% 是误差累积而非
   个别坏分子, 排除 2 个 drop=1.05% 的 conv 收益很小. **conv 全类 INT8 即可**.
2. **attn_out 的 2.25% drop 集中在 decoder**: 4 个 decoder attn_out (layers.2/3)
   drop 1.2~2.7%, 而 encoder 的 attn_out 全部 SAFE (drop=0%). decoder 注意力输出
   比 encoder 敏感.
3. **结论**: 即使发现 decoder attn_out 是坏分子, 排除它们只省 4 个小 Gemm, 部署
   复杂度却增加. 简化起见, attn_out 整类 FP16 仍是最稳选择.

## 量化策略

基于类别 + 逐层敏感度分析 + 算子级 profiling, 选定 **enc_full** 策略:
encoder 全 INT8 (conv+qkv+ffn+attn_out), decoder 仅 qkv INT8 (ffn/attn 敏感保 FP16).
注: conv 只存在于 vision_encoder (ResNet18), 无 decoder conv.

### 策略演进 (CPU EP, AVX-VNNI)

| 策略 | INT8 节点 | 大小 | turn match | ms/帧 | 加速 | 结论 |
|------|----------|------|-----------|-------|------|------|
| FP32 baseline | 0 | 193 MB | 100% | 21.6 | — | 参考 |
| conv+qkv | 53 | 75 MB | 99.1% | 16.0 | 快 26% | 原方案 |
| conv+qkv+enc_ffn | 61 | 67 MB | 98.6% | 13.5 | 快 37% | 加 encoder ffn |
| **enc_full (推荐)** | **65** | **66 MB** | **98.8%** | **12.9** | **快 40%** | **+encoder attn_out** |
| mixed (all ffn) | 69 | 54 MB | 85.3% | 12.1 | 快 44% | decoder ffn 崩 |

关键发现:
- **FFN 是最大瓶颈** (profiling 占 35% 时间), 量化 encoder ffn 收益最大 (快 2.5ms).
- **decoder ffn 敏感**: 全量化 ffn 掉到 85.3%, 因 ffn2 (3200->512) 是信息瓶颈,
  INT8 误差经残差放大. encoder ffn 安全 (输入来自 ResNet, 分布稳定).
- **encoder attn_out 全 SAFE** (逐层分析 drop=0%), decoder attn_out 有 4 个坏分子
  (layers.2/3), 故只量化 encoder 的 4 个.
- **加 encoder attn_out 增益小** (0.6ms), 因 attn_out 维度小; 但精度不降, 故纳入.

### 最终分层 (enc_full)

| 类别 | 量化方式 | 理由 |
|------|---------|------|
| Conv (21, 全 encoder) | INT8 | drop 1.95%, AVX-VNNI 加速 |
| QKV (32, enc+dec) | INT8 | drop 0.90%, 最不敏感 |
| encoder FFN1 (4) | INT8 | encoder 安全, profiling 最大瓶颈 |
| encoder FFN2 (4) | INT8 | encoder 安全 |
| encoder attn_out (4) | INT8 | 逐层分析全 SAFE (drop=0%) |
| decoder FFN1 (4) | FP16 | drop 5.71%, 敏感 |
| decoder FFN2 (4) | FP16 | drop 15.47%, 最敏感 |
| decoder attn_out (7) | FP16 | 4 个坏分子 (layers.2/3, drop 1.2~2.7%) |
| attn_core (22) | FP16 | softmax 附近动态范围大 |
| LayerNorm/Softmax/Erf/CumSum | FP32 | FP16 下方差/exp/累加溢出 |

两阶段流程:
1. **Stage 1**: 选定层静态 INT8 量化 (QDQ, per-channel 对称权重 + 非对称激活)
2. **Stage 2**: 剩余 FP32 部分 → FP16 (LayerNorm/Softmax/Erf/CumSum 保持 FP32)

生成: `scripts/quantize_mixed_onnx.py --preset enc_full` (preset 内置 encoder/decoder scope 过滤)

## 校准策略对比

### 方法

对 conv+qkv INT8 + FP16 的混合量化, 对比不同校准数据采样方式. 校准数据决定
INT8 的 scale/zero_point, 直接影响精度. (first-N 无意义: 按文件名排序前 N 帧
是视频开头连续帧, 不具代表性, 不纳入对比.)

脚本: `scripts/calib_compare.py`

```bash
.venv/bin/python scripts/calib_compare.py
```

### 结果 (666 帧, CPU EP)

| 策略 | 校准帧数 | turn match | drop | max_diff | avg |
|------|---------|-----------|------|----------|-----|
| **all (全量)** | **666** | **99.10%** | **+0.90%** | **0.002783** | **10.6ms** |
| uniform + thr=0.002 | 100 (从446帧中) | 98.95% | +1.05% | 0.002987 | 10.3ms |
| uniform + thr=0.0005 | 100 (从623帧中) | 98.50% | +1.50% | 0.002892 | 10.0ms |
| uniform + thr=0.001 | 100 (从560帧中) | 98.35% | +1.65% | 0.003496 | 10.5ms |
| uniform + thr=0.005 | 100 (从244帧中) | 98.35% | +1.65% | 0.002144 | 9.9ms |
| uniform-100 | 100 (从666帧中) | 97.90% | +2.10% | 0.002242 | 10.0ms |

### 结论

- **全量校准 (all-666) 最优**: turn_match 99.10%, drop 仅 0.90%. 全量数据给
  scale/zero_point 最完整的激活分布.
- **threshold 筛选无优势**: 阈值筛选 (排除边界帧) 并未提升精度, 反而因数据量
  减少而略差.
- **校准帧数越多越好**: uniform-100 (97.90%) < all-666 (99.10%), 差 1.2%.
- **推荐**: `--calib-mode all` (全量 666 帧校准). 量化只做一次, 校准多花几秒无所谓.
- 推理速度 (avg_ms) 各策略几乎相同 (~10ms), 因为模型结构一致, 仅 scale/zero_point 不同.

## 最终模型性能

### 生成

```bash
make quant-all   # quant-mixed + quant-verify
# 或手动:
.venv/bin/python scripts/quantize_mixed_onnx.py --preset enc_full --calib-mode all
# 输出: output/train/model_mixed_enc_full.onnx (65.7 MB)
```

### verify_results (666 帧, 对比 FP32 ONNX 参考, 纯 CPU)

```bash
make quant-verify
# 或:
.venv/bin/python scripts/batch_infer_onnx.py \
    --model output/train/model_mixed_enc_full.onnx \
    --output output/infer_results_mixed_enc_full.json
.venv/bin/python scripts/verify_results.py \
    --reference output/infer_results_onnx.json \
    --result output/infer_results_mixed_enc_full.json
```

结果:

| 指标 | 值 |
|------|-----|
| 模型大小 | 65.7 MB (FP32 原 202 MB, 压缩 3.1x) |
| turn match | **658/666 = 98.8%** |
| max left_vel diff | 0.003533 |
| max right_vel diff | 0.002510 |
| 推理速度 (CPU EP, 纯推理) | **~12.9 ms/帧** (FP32 21.6ms, 快 40%) |
| 端到端 (含图像预处理) | ~19 ms/帧 |
| 差异帧 | 8 帧, 全部 \|L-R\| < 0.001 (边界帧) |

差异帧都是左右轮速极接近的边界情况 (FP32 下 \|L-R\| < 0.001), 实际控制
意义下方向判断本就无意义, 不影响实际部署.

### 性能对比

| 模型 | 大小 | turn match | ms/帧 | 加速 | 备注 |
|------|------|-----------|-------|------|------|
| FP32 baseline | 202 MB | 100.0% | 21.6 | — | 参考 |
| Pure FP16 | 101 MB | 99.85% | 24.7 | 慢 14% | x86 CPU 无 FP16 单元, 每层需 Cast |
| conv+qkv (原方案) | 75 MB | 99.1% | 16.0 | 快 26% | |
| **enc_full (推荐)** | **66 MB** | **98.8%** | **12.9** | **快 40%** | **encoder 全 INT8** |
| mixed (all ffn) | 54 MB | 85.3% | 12.1 | 快 44% | decoder ffn 崩, 不可用 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/sensitivity_analysis.py` | 按类别敏感度分析 (5 类, 666 帧, CPU) |
| `scripts/per_layer_sensitivity.py` | 逐层 leave-one-in 敏感度 (conv + attn_out, 找坏分子) |
| `scripts/calib_compare.py` | 校准数据采样策略对比 (6 种策略) |
| `scripts/quantize_mixed_onnx.py` | 混合精度量化 (INT8 + FP16 两阶段, preset 式, 含 enc_full) |
| `scripts/batch_infer_onnx.py` | ONNX 批量推理 (生成 infer_results JSON) |
| `scripts/verify_results.py` | 对比两个 infer_results 的 turn match / max_diff |
| `tmp/sensitivity_per_class.json` | 按类别敏感度结果 (含每类完整 node 列表) |
| `tmp/sensitivity_per_layer.json` | 逐层敏感度结果 (含坏分子清单) |
| `tmp/calib_compare_results.json` | 校准策略对比结果 |
| `output/train/model_mixed_enc_full.onnx` | 最终混合精度模型 (65.7 MB, enc_full 策略) |
| `output/infer_results_mixed_enc_full.json` | 最终模型 666 帧推理结果 |
