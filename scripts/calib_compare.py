"""校准数据采样策略对比 (conv+qkv INT8 + FP16, 666 帧, CUDA 推理)

对 conv+qkv preset 的两阶段混合量化, 对比不同校准数据采样方式对精度的
影响. 旧文档 (docs/quantization.md) 的校准采样表数字无法复现, 本脚本
重新生成全部数据.

策略:
  first-N        : 排序后前 N 帧
  uniform-N      : 全量中均匀采样 N 帧
  all            : 全量 666 帧
  uniform+thr-X  : 先筛 |L-R| >= X 的帧, 再均匀采样 100 帧

用法:
    source scripts/cuda-env.sh
    .venv/bin/python scripts/calib_compare.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sensitivity_analysis import (
    CACHE_DIR, PREPROCESSED,
    load_action_denorm, load_images, load_refs, ensure_preprocessed,
    run_inference, compute_turn_match,
)
from quantize_mixed_onnx import (
    classify_nodes, make_calibration_reader, load_stats,
    _find_qdq_boundary_nodes,
)

CALIB_CACHE = CACHE_DIR / "calib_compare"
RESULTS_PATH = PROJECT_ROOT / "tmp" / "calib_compare_results.json"


def stage1_int8(out_path, calib_mode, calib_count, threshold):
    """conv+qkv INT8 量化, 复用 PREPROCESSED 缓存."""
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    m = onnx.load(str(PREPROCESSED), load_external_data=False)
    cls = classify_nodes(m)
    nodes = sorted(cls["conv"]) + sorted(cls["qkv"])
    reader = make_calibration_reader(calib_mode, count=calib_count, threshold=threshold)
    quantize_static(
        str(PREPROCESSED), str(out_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=nodes,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
            "EnableSubgraph": True,
        },
    )
    return nodes


def stage2_fp16(stage1_path, out_path, int8_nodes):
    """剩余 FP32 -> FP16, LayerNorm/Softmax/Erf/CumSum 保持 FP32."""
    from onnxconverter_common import float16
    m = onnx.load(str(stage1_path))
    qdq_boundary = _find_qdq_boundary_nodes(m)
    node_block_list = list(set(int8_nodes + qdq_boundary))
    m = float16.convert_float_to_float16(
        m, keep_io_types=True, disable_shape_infer=True,
        op_block_list=[
            "LayerNormalization", "Softmax", "Erf", "ReduceMean",
            "QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
            "MatMulInteger", "GemmInteger", "ConvInteger",
        ],
        node_block_list=node_block_list,
    )
    onnx.save(m, str(out_path))


# 要对比的校准策略 (first-N 无意义: 按文件名排序前 N 帧是视频开头连续帧, 不具代表性)
STRATEGIES = [
    ("uniform-100",        {"mode": "uniform", "count": 100, "threshold": 0.0}),
    ("all-666",            {"mode": "all",     "count": 666, "threshold": 0.0}),
    ("uniform+thr-0.0005", {"mode": "uniform", "count": 100, "threshold": 0.0005}),
    ("uniform+thr-0.001",  {"mode": "uniform", "count": 100, "threshold": 0.001}),
    ("uniform+thr-0.002",  {"mode": "uniform", "count": 100, "threshold": 0.002}),
    ("uniform+thr-0.005",  {"mode": "uniform", "count": 100, "threshold": 0.005}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    CALIB_CACHE.mkdir(parents=True, exist_ok=True)
    state_np = load_stats()
    a_q01, a_d = load_action_denorm()
    img_tensors = load_images(args.frames)
    refs = load_refs(len(img_tensors))
    print(f"[eval] {len(img_tensors)} frames", flush=True)

    ensure_preprocessed()

    results = {}
    t_start = time.time()
    for i, (tag, cfg) in enumerate(STRATEGIES):
        elapsed = time.time() - t_start
        print(f"\n[{i+1}/{len(STRATEGIES)}] {tag}  elapsed={elapsed:.0f}s", flush=True)
        final = CALIB_CACHE / f"mixed_{tag}.onnx"
        if not final.exists() or args.no_cache:
            stage1 = CALIB_CACHE / f"stage1_{tag}.onnx"
            t_q = time.time()
            int8_nodes = stage1_int8(stage1, cfg["mode"], cfg["count"], cfg["threshold"])
            print(f"  stage1 (INT8 {len(int8_nodes)} nodes) in {time.time()-t_q:.1f}s", flush=True)
            t_f = time.time()
            stage2_fp16(stage1, final, int8_nodes)
            print(f"  stage2 (FP16) in {time.time()-t_f:.1f}s", flush=True)
            stage1.unlink(missing_ok=True)
        else:
            print(f"  [cache hit]", flush=True)

        try:
            actions, avg_ms = run_inference(final, img_tensors, state_np)
        except Exception as e:
            print(f"  [skip] inference failed: {e}", flush=True)
            continue
        tm, md = compute_turn_match(actions, refs, a_q01, a_d)
        drop = 100.0 - tm
        print(f"  -> turn_match={tm:.2f}%  drop={drop:+.2f}%  "
              f"max_diff={md:.6f}  avg={avg_ms:.1f}ms", flush=True)
        results[tag] = {
            "calib_mode": cfg["mode"],
            "calib_count": cfg["count"],
            "threshold": cfg["threshold"],
            "turn_match_pct": round(tm, 4),
            "drop_pct": round(drop, 4),
            "max_diff": md,
            "avg_ms": avg_ms,
            "size_mb": final.stat().st_size / 1e6,
        }

    # 汇总 (按 turn_match 降序, 最优在前)
    print(f"\n{'='*90}")
    print("校准策略对比 (conv+qkv INT8 + FP16, 666 帧):")
    print(f"{'strategy':<22} {'turn%':>8} {'drop':>8} {'max_diff':>10} {'avg_ms':>8} {'size_MB':>8}")
    print(f"{'-'*90}")
    sorted_res = sorted(results.items(), key=lambda x: -x[1]["turn_match_pct"])
    for tag, v in sorted_res:
        print(f"{tag:<22} {v['turn_match_pct']:>8.2f} {v['drop_pct']:>+8.2f} "
              f"{v['max_diff']:>10.6f} {v['avg_ms']:>8.1f} {v['size_mb']:>8.1f}")

    best = sorted_res[0]
    print(f"\n[best] {best[0]}: turn_match={best[1]['turn_match_pct']:.2f}%, "
          f"drop={best[1]['drop_pct']:+.2f}%, max_diff={best[1]['max_diff']:.6f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] -> {RESULTS_PATH}", flush=True)
    print(f"[total] {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
