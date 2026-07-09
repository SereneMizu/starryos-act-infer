"""
ACT 模型混合精度量化 (INT8 + FP16)

策略:
  Stage 1 - 静态 INT8 量化计算密集的大层 (Conv / FFN / QKV / Gemm):
            节省 MAC, 利用 AVX-VNNI / ARM NEON DOT / RISC-V V 扩展
  Stage 2 - 把剩余 FP32 (LayerNorm/Softmax/Erf/attention core/小MatMul/中间激活)
            全部转换为 FP16: 内存减半, 嵌入式平台 (SG2002 256MB) 必需

精度敏感的 attention core (Q@K^T, attn@V) 和 action_head 只做 FP16, 不做 INT8,
避免 softmax 前后大动态范围导致的精度崩溃。

用法:
    .venv/bin/python scripts/quantize_mixed_onnx.py                       # 默认 mixed
    .venv/bin/python scripts/quantize_mixed_onnx.py --preset conservative # 仅 Conv+FFN 做 INT8
    .venv/bin/python scripts/quantize_mixed_onnx.py --preset aggressive   # Conv+FFN+QKV+Gemm 做 INT8
    .venv/bin/python scripts/quantize_mixed_onnx.py --no-fp16             # 关闭 stage 2 (仅 INT8)
    .venv/bin/python scripts/quantize_mixed_onnx.py --dynamic             # 动态权重量化 (无 calibration)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FP32 = PROJECT_ROOT / "output" / "train" / "model.onnx"
STATS_PATH = PROJECT_ROOT / "output" / "dataset" / "meta" / "stats.json"
IMG_DIR = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"
REF_PATH = PROJECT_ROOT / "output" / "infer_results_onnx.json"
CALIB_COUNT = 100

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_stats() -> np.ndarray:
    with open(STATS_PATH) as f:
        raw = json.load(f)
    state_q01 = np.array(raw["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.array(raw["observation.state"]["q99"], dtype=np.float32)
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_normed = (2 * (np.zeros_like(state_q01) - state_q01) / d - 1).reshape(1, -1).astype(np.float32)
    return state_normed


def load_frame_diffs() -> list[float]:
    """从参考结果中读取每帧的 |left_vel - right_vel|, 用于阈值筛选校准图像"""
    with open(REF_PATH) as f:
        refs = json.load(f)
    return [abs(r["left_vel"] - r["right_vel"]) for r in refs]


def select_calibration_images(mode: str = "first", count: int = CALIB_COUNT,
                               threshold: float = 0.0) -> list[Path]:
    """选择校准图像. mode 可选:
       'first'     - 排序后前 N 张
       'uniform'   - 从全部图像中等间隔采样 N 张
       'all'       - 全部图像
       当 threshold > 0 时, 先筛选 |L-R| >= threshold 的帧, 再在子集中采样.
    """
    all_images = sorted(IMG_DIR.glob("*.jpg"))
    if threshold > 0:
        diffs = load_frame_diffs()
        filtered = [(p, d) for p, d in zip(all_images, diffs) if d >= threshold]
        if not filtered:
            print(f"[warn] threshold={threshold} 无帧通过, 回退到全部")
        else:
            all_images = [p for p, _ in filtered]
            print(f"[calib] threshold={threshold} 通过 {len(all_images)}/{len(diffs)} 帧")
    total = len(all_images)
    if mode == "all":
        return all_images
    if mode == "uniform":
        if count >= total:
            return all_images
        idxs = np.linspace(0, total - 1, count, dtype=int)
        return [all_images[i] for i in idxs]
    # default: first
    return all_images[:count]


def make_calibration_reader(mode: str = "first", count: int = CALIB_COUNT, threshold: float = 0.0):
    from onnxruntime.quantization import CalibrationDataReader

    state_np = load_stats()
    images = select_calibration_images(mode, count, threshold)

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.idx = 0

        def get_next(self) -> Optional[dict]:
            if self.idx >= len(images):
                return None
            img = Image.open(images[self.idx]).convert("RGB")
            tensor = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
            self.idx += 1
            return {"images": tensor, "state": state_np}

        def rewind(self) -> None:
            self.idx = 0

    reader = _Reader()
    # expose image count for logging
    reader.num_images = len(images)  # type: ignore[attr-defined]
    return reader


def classify_nodes(model: onnx.ModelProto):
    """基于预处理后的图分类节点.

    quant_pre_process 会把 (MatMul + Add) / (MatMul + bias) 融合成 Gemm,
    命名为 `.../MatMulAddFusion`. 所以 FFN linear1/2 和 QKV 投影 (原 MatMul)
    都变成了 Gemm; 注意力输出投影本来就是 Gemm.

    返回 dict: 类别名 -> set(节点名)。类别:
      conv            : ResNet18 卷积 (21 个)
      ffn1            : Transformer FFN linear1 (升维 512->3200, Gemm, 8 个)
      ffn2            : Transformer FFN linear2 (降维 3200->512, Gemm, 8 个)
      qkv             : 注意力 Q/K/V 投影 (Gemm, MatMulAddFusion, 32 个)
      attn_out_proj   : 注意力输出投影 Gemm (11 个)
      attn_core       : Q@K^T (MatMul_3) + attn@V (MatMul_4), softmax 附近, 仍为 MatMul
      tiny            : action_head / state_encoder / latent_proj, 维度极小

    INT8 候选: conv, ffn1, ffn2, qkv, attn_out_proj (由 preset 决定)
    永远不量化: attn_core, tiny (走 FP16)
    """
    classes = {
        "conv": set(),
        "ffn1": set(),
        "ffn2": set(),
        "qkv": set(),
        "attn_out_proj": set(),
        "attn_core": set(),
        "tiny": set(),
    }

    for n in model.graph.node:
        name, op = n.name, n.op_type
        if op not in ("Conv", "MatMul", "Gemm"):
            continue

        # tiny 永远不量化
        if "action_head" in name or "state_encoder" in name or "latent_proj" in name:
            classes["tiny"].add(name)
            continue

        # attention core: Q@K^T (MatMul_3) + attn@V (MatMul_4), 保留 MatMul 形式
        if op == "MatMul" and (name.endswith("/MatMul_3") or name.endswith("/MatMul_4")):
            classes["attn_core"].add(name)
            continue

        # Conv: ResNet18
        if op == "Conv":
            classes["conv"].add(name)
            continue

        # FFN: Gemm, 名称含 linear1 或 linear2 (拆开以便敏感度分析)
        if op == "Gemm" and "linear1" in name:
            classes["ffn1"].add(name)
            continue
        if op == "Gemm" and "linear2" in name:
            classes["ffn2"].add(name)
            continue

        # 注意力输出投影: Gemm, 名称以 self_attn/Gemm 或 multihead_attn/Gemm 结尾
        if op == "Gemm" and (name.endswith("/self_attn/Gemm") or name.endswith("/multihead_attn/Gemm")):
            classes["attn_out_proj"].add(name)
            continue

        # QKV 投影: Gemm (preprocess 融合后), 名称含 self_attn/MatMul*/MatMulAddFusion 或 multihead_attn/MatMul*/MatMulAddFusion
        if op == "Gemm" and "MatMulAddFusion" in name and (
            "/self_attn/" in name or "/multihead_attn/" in name
        ):
            classes["qkv"].add(name)
            continue

    return classes


# 每个 preset 决定哪些类别的层走 INT8.
# 值的含义: "no"=不量化, "all"=全量化, "encoder"=仅 encoder 部分, "decoder"=仅 decoder 部分.
#   (向后兼容: True -> "all", False -> "no", 在 _normalize_preset 里转换)
# ffn = ffn1 + ffn2
PRESETS = {
    "conv-only":       {"conv": "all",  "ffn1": "no", "ffn2": "no", "qkv": "no", "attn_out_proj": "no"},
    "ffn1-only":       {"conv": "no",  "ffn1": "all", "ffn2": "no", "qkv": "no", "attn_out_proj": "no"},
    "ffn2-only":       {"conv": "no",  "ffn1": "no", "ffn2": "all", "qkv": "no", "attn_out_proj": "no"},
    "qkv-only":        {"conv": "no",  "ffn1": "no", "ffn2": "no", "qkv": "all", "attn_out_proj": "no"},
    "attn-out-only":   {"conv": "no",  "ffn1": "no", "ffn2": "no", "qkv": "no", "attn_out_proj": "all"},
    "ffn-only":        {"conv": "no",  "ffn1": "all", "ffn2": "all", "qkv": "no", "attn_out_proj": "no"},
    "conv+qkv":        {"conv": "all",  "ffn1": "no", "ffn2": "no", "qkv": "all", "attn_out_proj": "no"},
    "conv+attn":       {"conv": "all",  "ffn1": "no", "ffn2": "no", "qkv": "no", "attn_out_proj": "all"},
    "conv+qkv+attn":   {"conv": "all",  "ffn1": "no", "ffn2": "no", "qkv": "all", "attn_out_proj": "all"},
    "conservative":    {"conv": "all",  "ffn1": "all", "ffn2": "all", "qkv": "no", "attn_out_proj": "no"},
    "conv+ffn2":       {"conv": "all",  "ffn1": "no", "ffn2": "all", "qkv": "no", "attn_out_proj": "no"},
    "conv+ffn1":       {"conv": "all",  "ffn1": "all", "ffn2": "no", "qkv": "no", "attn_out_proj": "no"},
    "mixed":           {"conv": "all",  "ffn1": "all", "ffn2": "all", "qkv": "all", "attn_out_proj": "no"},
    "aggressive":      {"conv": "all",  "ffn1": "all", "ffn2": "all", "qkv": "all", "attn_out_proj": "all"},
    # encoder 全 INT8 (conv+qkv+ffn+attn_out), decoder 仅 qkv INT8.
    # (conv 只存在于 vision_encoder/ResNet18, 无 decoder conv.)
    # decoder ffn 敏感 (ffn2 drop 15.47%), decoder attn_out 有 4 个坏分子 (layers.2/3).
    # CPU 12.9ms/帧 (FP32 21.6ms, 快 40%), 66MB, turn match 98.8%. 详见 docs/quantization_v2.md.
    "enc_full":        {"conv": "all",  "ffn1": "encoder", "ffn2": "encoder", "qkv": "all", "attn_out_proj": "encoder"},
}


def _is_encoder_node(name: str) -> bool:
    """encoder = vision_encoder + encoder transformer layers. decoder = decoder layers."""
    return "/decoder/" not in name


def _filter_by_scope(names, scope: str) -> list[str]:
    """按 scope 过滤节点名列表. scope: 'all'/'encoder'/'decoder'/'no'."""
    if scope == "no":
        return []
    if scope == "all":
        return sorted(names)
    if scope == "encoder":
        return sorted(n for n in names if _is_encoder_node(n))
    if scope == "decoder":
        return sorted(n for n in names if not _is_encoder_node(n))
    return []


def stage1_int8_quantize(
    model_in: Path,
    model_out: Path,
    preset: str,
    calibrate_method: str = "MinMax",
    calib_mode: str = "uniform",
    threshold: float = 0.0,
    calib_count: int = 100,
) -> list[str]:
    """Stage 1: 静态 INT8 量化选定层 (QDQ 格式, per-channel 对称权重 + 非对称激活).

    calibrate_method: 'MinMax' (快, 默认) 或 'Entropy' (慢但更准, 适合 attention 模型).
    calib_mode: 校准图像采样方式.
    threshold: 仅选择 |L-R| >= threshold 的帧 (0=不筛选).
    返回被量化的节点名列表 (供 stage2 FP16 转换时跳过这些节点)。"""
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static, shape_inference,
    )

    method_map = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    calib = method_map.get(calibrate_method, CalibrationMethod.MinMax)

    preprocessed = model_in.with_suffix(".preprocessed.onnx")
    shape_inference.quant_pre_process(str(model_in), str(preprocessed))

    m = onnx.load(str(preprocessed), load_external_data=False)
    cls = classify_nodes(m)
    cfg = PRESETS[preset]

    nodes_to_quantize: list[str] = []
    for cat in ("conv", "ffn1", "ffn2", "qkv", "attn_out_proj"):
        nodes_to_quantize += _filter_by_scope(cls[cat], cfg[cat])

    def _cnt(cat):
        return len(_filter_by_scope(cls[cat], cfg[cat]))

    print(f"[stage1:{preset}] method={calibrate_method} INT8 nodes={len(nodes_to_quantize)}  "
          f"(conv={_cnt('conv')}, ffn1={_cnt('ffn1')}, ffn2={_cnt('ffn2')}, "
          f"qkv={_cnt('qkv')}, attn_out_proj={_cnt('attn_out_proj')})")
    print(f"[stage1:{preset}] skip (FP16 path): "
          f"attn_core={len(cls['attn_core'])}, tiny={len(cls['tiny'])}")

    reader = make_calibration_reader(calib_mode, count=calib_count, threshold=threshold)
    print(f"[stage1:{preset}] calibration on {reader.num_images} images (mode={calib_mode})...")

    quantize_static(
        str(preprocessed),
        str(model_out),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=calib,
        nodes_to_quantize=nodes_to_quantize,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
            "EnableSubgraph": True,
        },
    )
    preprocessed.unlink(missing_ok=True)
    return nodes_to_quantize


def _find_qdq_boundary_nodes(model: onnx.ModelProto) -> list[str]:
    """找出所有"夹在 QDQ 节点之间"的节点 —— 它们的输入来自 DequantizeLinear,
    或输出被 QuantizeLinear 消费. 这些节点 (Relu, Add, MaxPool, ...) 必须保持
    FP32, 否则 FP16 转换会破坏 INT8 路径的类型一致性。"""
    # 收集所有 DQ 输出和 Q 输入
    dq_outputs = set()
    q_inputs = set()
    for n in model.graph.node:
        if n.op_type == "DequantizeLinear":
            dq_outputs.update(n.output)
        elif n.op_type == "QuantizeLinear":
            q_inputs.update(n.input)

    boundary = []
    for n in model.graph.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear", "Constant"):
            continue
        # 输入来自 DQ 输出, 或输出被 Q 消费
        if any(inp in dq_outputs for inp in n.input) or any(out in q_inputs for out in n.output):
            boundary.append(n.name)
    return boundary


def stage2_fp16_convert(
    model_in: Path,
    model_out: Path,
    int8_nodes: list[str],
    keep_io_types: bool = True,
) -> None:
    """Stage 2: 把 stage1 没量化的 FP32 部分转 FP16.

    关键: 除了已 INT8 量化的节点本身, 还要把夹在 QDQ 节点之间的中间节点
    (Relu/Add/MaxPool/Concat 等) 也加入 node_block_list, 否则它们的输出
    会被错误标成 FP16, 而下游 QuantizeLinear 期望 FP32 输入。

    op_block_list 里的算子保持 FP32 内部计算, 避免 LayerNorm/Softmax/Erf
    在纯 FP16 下精度崩溃。"""
    from onnxconverter_common import float16

    m = onnx.load(str(model_in))
    qdq_boundary = _find_qdq_boundary_nodes(m)
    node_block_list = list(set(int8_nodes + qdq_boundary))

    print(f"[stage2] node_block_list={len(node_block_list)} "
          f"(int8={len(int8_nodes)}, qdq_boundary={len(qdq_boundary)})")

    m = float16.convert_float_to_float16(
        m,
        keep_io_types=keep_io_types,
        disable_shape_infer=True,
        op_block_list=[
            "LayerNormalization",  # 方差计算 FP16 会溢出
            "Softmax",              # exp 易溢出
            "Erf",                  # GELU 在小值域 FP16 精度差
            "ReduceMean",           # LayerNorm 分解后用到的
            "QuantizeLinear",       # INT8 路径, 不能转
            "DequantizeLinear",
            "DynamicQuantizeLinear",
            "MatMulInteger",
            "GemmInteger",
            "ConvInteger",
        ],
        node_block_list=node_block_list,
    )
    onnx.save(m, str(model_out))


def quantize_dynamic_only(model_in: Path, model_out: Path, op_types: list[str]) -> None:
    """动态权重量化 (无 calibration, 只量化权重, 运行时动态量化激活)"""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    print(f"[dynamic] op_types={op_types}")
    quantize_dynamic(
        str(model_in),
        str(model_out),
        op_types_to_quantize=op_types,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MODEL_FP32)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="enc_full",
                        help="量化策略 (默认 enc_full: encoder 全 INT8 + decoder 仅 qkv)")
    parser.add_argument("--calib", choices=["MinMax", "Entropy", "Percentile"], default="MinMax",
                        help="calibration 方法 (Entropy 内存占用高, 60+ 节点 + 全量校准易 OOM)")
    parser.add_argument("--calib-mode", choices=["first", "uniform", "all"], default="all",
                        help="校准图像采样方式: all(全量666, 默认最优), uniform(均匀N), first(排序前N)")
    parser.add_argument("--calib-count", type=int, default=100,
                        help="校准帧数量 (仅对 uniform/first 有效, 默认100)")
    parser.add_argument("--calib-threshold", type=float, default=0.0,
                        help="仅选择 |L-R| >= threshold 的帧做校准 (0=不筛选)")
    parser.add_argument("--no-fp16", action="store_true",
                        help="跳过 stage2 FP16 转换 (结果只有 INT8, 其余 FP32)")
    parser.add_argument("--dynamic", action="store_true",
                        help="用动态权重量化 (忽略 preset 和 fp16)")
    parser.add_argument("--dynamic-ops", type=str, default="MatMul,Gemm")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[error] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"[input] {args.input} ({args.input.stat().st_size / 1e6:.1f} MB)")

    if args.output is None:
        if args.dynamic:
            args.output = args.input.with_name(f"{args.input.stem}_dynamic_int8.onnx")
        elif args.no_fp16:
            args.output = args.input.with_name(f"{args.input.stem}_int8_{args.preset}.onnx")
        else:
            args.output = args.input.with_name(f"{args.input.stem}_mixed_{args.preset}.onnx")

    if args.dynamic:
        ops = [s.strip() for s in args.dynamic_ops.split(",") if s.strip()]
        quantize_dynamic_only(args.input, args.output, ops)
    else:
        # Stage 1: INT8
        stage1_out = args.output.with_suffix(".stage1.onnx") if not args.no_fp16 else args.output
        int8_nodes = stage1_int8_quantize(args.input, stage1_out, args.preset, args.calib, args.calib_mode, args.calib_threshold, args.calib_count)

        if not args.no_fp16:
            # Stage 2: FP16 剩余部分
            print(f"[stage2] converting remaining FP32 -> FP16 ...")
            stage2_fp16_convert(stage1_out, args.output, int8_nodes, keep_io_types=True)
            stage1_out.unlink(missing_ok=True)

    out_mb = args.output.stat().st_size / 1e6
    print(f"[output] {args.output} ({out_mb:.1f} MB)")

    m = onnx.load(str(args.output))
    onnx.checker.check_model(m)
    print("[output] onnx check passed")


if __name__ == "__main__":
    main()
