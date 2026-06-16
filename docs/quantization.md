# ONNX 混合精度量化

## 概览

对 ACT 模型的 ONNX 推理做 INT8 + FP16 混合精度量化，在保证推理正确性的前提下降低模型体积并加速推理。

整体流程：

```
FP32 ONNX → 逐层敏感度分析 → 确定量化策略 → Stage1: 选定层 INT8 静态量化 → Stage2: 剩余层 FP16 转换 → 混合精度 ONNX
```

## 模型结构

ACT 模型由 ResNet18 视觉编码器 + Transformer 编码器/解码器组成，ONNX 导出后共 677 个节点：

| 模块 | 算子类型 | 数量 | 权重维度 | 说明 |
|------|---------|------|---------|------|
| ResNet18 Conv | Conv | 21 | 64~512 通道 | 视觉特征提取 |
| FFN linear1 | Gemm (预处理后) | 8 | 512→3200 | Transformer FFN 升维 |
| FFN linear2 | Gemm (预处理后) | 8 | 3200→512 | Transformer FFN 降维 |
| QKV 投影 | Gemm (预处理后) | 32 | 512→512 | 注意力 Q/K/V 投影 |
| 注意力输出投影 | Gemm | 11 | 512→512 | self_attn + multihead_attn out_proj |
| 注意力核心 | MatMul | 16 | (无权重) | Q@K^T (MatMul_3) + attn@V (MatMul_4) |
| LayerNorm | LayerNormalization | 22 | — | 归一化 |
| Softmax | Softmax | 12 | — | 注意力权重 |
| GELU | Erf | 9 | — | FFN 激活函数 |
| action_head | MatMul | 1 | 512→3 | 最终动作输出（极小） |

> `quant_pre_process` 会把 FFN 和 QKV 的 `MatMul + Add` 融合成 `Gemm`（命名为 `.../MatMulAddFusion`），所以预处理后的图中这些层都是 Gemm。

## 逐层敏感度分析

### 方法

不依赖经验假设，对每一类层**单独**做 INT8 静态量化（其他层保持 FP32），用 ORT Python API 跑 100 帧评估，测量 turn match 下降幅度。

```bash
.venv/bin/python scripts/sensitivity_analysis.py
```

### 结果

基准：FP32 模型 turn match = 100%

| 量化的层类别 | INT8 节点数 | avg (ms) | turn match | turn drop | max diff | 敏感度 |
|-------------|-----------|---------|-----------|-----------|---------|--------|
| conv-only | 21 | 17.0 | 100% | 0% | 0.0018 | **不敏感** |
| qkv-only | 32 | 19.8 | 100% | 0% | 0.0003 | **不敏感** |
| attn-out-only | 11 | 20.2 | 95% | 5% | 0.0020 | 中等 |
| ffn1-only | 8 | 19.3 | 89% | 11% | 0.0025 | 敏感 |
| ffn2-only | 8 | 18.5 | 84% | 16% | 0.0024 | **最敏感** |

### 关键发现

**FFN（尤其是 linear2）是最敏感的层**，与直觉相反。原因：

- FFN 的输出通过残差连接直接影响后续所有层的输入
- FFN linear2（3200→512）是信息瓶颈，量化误差在此处被放大到整个 hidden_dim
- FFN linear1 的输出经过 GELU 激活，INT8 量化在 GELU 的小值区域（接近 0 的线性段）精度较差

**Conv 和 QKV 对量化完全不敏感**（turn drop = 0%）：
- ResNet18 Conv 有成熟的 INT8 量化经验，且 per-channel 量化足够精确
- QKV 投影是线性变换，权重分布均匀，INT8 损失极小

## 最终量化策略

基于敏感度分析，选择 **conv + qkv → INT8，其余 → FP16**：

| 层类别 | 量化方式 | 理由 |
|--------|---------|------|
| Conv (21) | INT8 | 不敏感 + x86 上有 AVX-VNNI 加速 |
| QKV (32) | INT8 | 不敏感，省内存 |
| FFN linear1/2 (16) | **FP16** | 最敏感，不能 INT8 |
| 注意力输出投影 (11) | FP16 | 中等敏感 |
| 注意力核心 (16) | FP16 | softmax 附近，动态范围大 |
| LayerNorm/Softmax/Erf | FP32 内部计算 | FP16 下方差/exp 会溢出 |

## 校准数据采样方式对比

校准图像的选择方式直接影响 INT8 量化的 scale/zero_point 精度。

### 方法

对比以下策略（`mixed` preset, Conv+FFN+QKV→INT8, 53节点，评估全量 666 帧）：

| 策略 | 校准帧筛选条件 | 采帧方式 | 校准数 |
|------|--------------|---------|-------|
| uniform (基线) | 全部帧 | 均匀采样 | 100 |
| all (基线) | 全部帧 | 全部 | 666 |
| first (基线) | 全部帧 | 排序前N | 100 |
| **阈值筛选** | `\|L-R\| ≥ threshold` | 从通过帧中均匀采样 | 100 |

### 结果

测试环境：i5-13500H, ORT 1.26.0, 666 帧

| 策略 | 校准帧数 | turn match | speedup |
|------|---------|-----------|---------|
| FP32 baseline | — | 100.0% | 1.00x |
| uniform (基线) | 100 (全量均匀) | 89.2% | 1.63x |
| first (基线) | 100 (前30%) | 89.0% | 1.64x |
| all (基线) | 666 (全量) | 86.6% | 1.45x |
| **uniform threshold=0.001** | **100 (从560帧中采)** | **92.3%** | **1.66x** |
| uniform threshold=0.0005 | 100 (从623帧中采) | 86.3% | 1.55x |
| uniform threshold=0.002 | 100 (从446帧中采) | 86.0% | 1.57x |
| uniform threshold=0.005 | 100 (从244帧中采) | 86.5% | 1.67x |

### 结论

- **阈值 0.001 均匀采样最优**（92.3%），显著高于全量统一采样（89.2%）
- 排除最模糊的 16% 帧（\|L-R\| < 0.001）可去除校准噪声，让 scale/zero_point 聚焦于有意义的激活范围
- 阈值再提高（0.002/0.005）反而变差——排除帧过多，丢失大量数据分布
- 校准帧数量（100/200/400）对精度无影响——三种数量下 turn match 均为 92.3%，说明只要做了阈值筛选+均匀采样，少量帧即可覆盖分布
- **推荐校准策略：`--calib-mode uniform --calib-threshold 0.001 --calib-count 100`**（100 帧足够，量化更快）

## 两阶段量化流程

### Stage 1: INT8 静态量化

```python
quantize_static(
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,   # 非对称激活
    weight_type=QuantType.QInt8,        # 对称权重 + per-channel
    per_channel=True,
    nodes_to_quantize=[conv_nodes + qkv_nodes],  # 只量化 Conv + QKV
)
```

- QDQ 格式（QuantizeLinear/DequantizeLinear 节点），兼容性好
- per-channel 量化权重，每个输出通道独立 scale
- 200 张校准图片做 MinMax 校准（`--calib-mode uniform`，均匀采样推荐）

### Stage 2: FP16 转换

```python
float16.convert_float_to_float16(
    op_block_list=["LayerNormalization", "Softmax", "Erf",
                   "QuantizeLinear", "DequantizeLinear", ...],
    node_block_list=int8_nodes + qdq_boundary_nodes,
)
```

关键处理：
- **node_block_list** 包含所有已 INT8 量化的节点 + 夹在 QDQ 之间的中间节点（Relu/Add/MaxPool 等），避免破坏 INT8 路径的类型一致性
- **op_block_list** 让 LayerNorm/Softmax/Erf 保持 FP32 内部计算，避免 FP16 精度崩溃

## 使用方法

### 生成混合精度模型

```bash
# 推荐：conv + qkv → INT8, 其余 → FP16 (阈值筛选+均匀采样校准)
.venv/bin/python scripts/quantize_mixed_onnx.py --preset conv+qkv --calib-threshold 0.001

# 等价于默认值:
#   --calib-mode uniform --calib-count 100 --calib-threshold 0.001

# 输出: output/train/model_mixed_conv+qkv.onnx
```

### 可用 preset

| preset | INT8 层 | FP16 层 | 模型大小 | 适用场景 |
|--------|--------|---------|---------|---------|
| conv+qkv | Conv + QKV | FFN + Attn + 其他 | 79.5 MB | **推荐**，精度/速度/体积平衡 |
| conv-only | Conv | FFN + QKV + Attn + 其他 | 87.6 MB | 精度最高，速度提升有限 |
| aggressive | Conv + FFN + QKV + AttnOut | 少量 | 51.0 MB | 体积最小，精度有损失 |
| conservative | Conv + FFN | QKV + Attn + 其他 | 61.9 MB | FFN 量化导致精度下降 |

### 敏感度分析

```bash
.venv/bin/python scripts/sensitivity_analysis.py
```

### Benchmark

```bash
# 用 Rust ORT 程序跑完整 666 帧
docker exec starryos-act-infer /workspace/act-infer-ort/target/release/act-infer-ort \
    --model /workspace/output/train/model_mixed_conv+qkv.onnx \
    --dir /workspace/output/dataset/videos/observation.images.fpv/chunk-000 \
    --stats /workspace/output/dataset/meta/stats.json \
    --reference /workspace/output/infer_results_onnx.json
```

## 性能对比

测试环境：i5-13500H (AVX-VNNI), ONNX Runtime 1.26.0, 666 帧

| 模型 | 大小 | avg | speedup | turn match | max diff |
|------|------|-----|---------|-----------|----------|
| FP32 baseline | 202 MB | 19.7ms | 1.00x | 99.1% | 0.001 |
| Pure FP16 | 101 MB | 22.1ms | 0.89x | 99.7% | 0.001 |
| INT8 (全量化, 旧) | 51 MB | 35.7ms | 0.55x | 72.4% | 0.016 |
| **Conv+QKV INT8 + FP16** | **79.5 MB** | **14.2ms** | **1.39x** | **97.7%** | **0.002** |

> Pure FP16 在 x86 CPU 上比 FP32 慢——x86 没有 FP16 计算单元，需转换到 FP32 计算。但在嵌入式平台（SG2002 256MB），FP16 的内存减半是刚需。

> 差异帧绝大多数是左右轮速接近的边界情况（|L-R| < 0.002），实际控制意义下方向判断无意义。

## 跨平台考虑

| 平台 | 推荐 | 原因 |
|------|------|------|
| x86 CPU (QEMU) | Conv+QKV INT8 + FP16 | AVX-VNNI 加速 INT8，FP16 有模拟开销但可接受 |
| SG2002 (256MB) | Conv+QKV INT8 + FP16 | 内存受限，FP16 权重减半是刚需 |
| RK3588 NPU | 用 RKNN 工具链 | NPU 原生支持 INT8，走单独的 rknn_compile.py 流程 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/quantize_mixed_onnx.py` | 混合精度量化脚本（INT8 + FP16 两阶段） |
| `scripts/sensitivity_analysis.py` | 逐层敏感度分析脚本 |
| `output/train/model_mixed_conv+qkv.onnx` | 推荐的混合精度模型 (79.5 MB) |
