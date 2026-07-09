"""逐层 leave-one-in INT8 敏度分析 (666 帧, CPU 推理)

针对按类别分析中"不确定"的类别 (conv drop 1.95%, attn_out drop 2.25%),
对每个层单独做 INT8 (其余 FP32), 找出类内的"坏分子": 哪几个层导致整体
drop 偏高. 明确的类别不测:
  - qkv (drop 0.9%)   : 全类安全, 无需逐层
  - ffn1 (drop 5.7%)  : 全类敏感, 直接 FP16
  - ffn2 (drop 15.5%) : 全类敏感, 直接 FP16

## 标识方式

直接用 ONNX 节点全名 (node_name) 作为 key. 它就是图中该节点的名字, 跨次
运行天然稳定. 量化脚本可直接拿 node_name 喂给 nodes_to_quantize.

## leave-one-in 的含义

只把目标层量化为 INT8, 其他全部 FP32. turn_drop 反映该层自身对 INT8 的
固有敏感度. (注意: 组合量化时误差会叠加, 最终策略仍需跑组合验证.)

用法:
    .venv/bin/python scripts/per_layer_sensitivity.py                       # 默认 conv + attn_out
    .venv/bin/python scripts/per_layer_sensitivity.py --categories conv qkv # 指定类别
    .venv/bin/python scripts/per_layer_sensitivity.py --frames 100          # 快速测试
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

# 复用按类别脚本的公共函数 + 缓存路径
from sensitivity_analysis import (
    CACHE_DIR, PREPROCESSED,
    load_action_denorm, load_images, load_refs, ensure_preprocessed,
    run_inference, compute_turn_match,
)
from quantize_mixed_onnx import classify_nodes, make_calibration_reader, load_stats

# classify_nodes 返回的类别名
ALL_CATEGORIES = ["conv", "qkv", "attn_out_proj", "ffn1", "ffn2"]
# 默认只测"中间地带"的类别 (按类别分析 drop 在 1~3% 之间)
DEFAULT_CATEGORIES = ["conv", "attn_out_proj"]


def quantize_one_layer(node_name, out_path, calib_count=50):
    """leave-one-in: 只把 node_name 量化为 INT8, 其余保持 FP32."""
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    reader = make_calibration_reader("uniform", count=calib_count, threshold=0.0)
    quantize_static(
        str(PREPROCESSED), str(out_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=[node_name],
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
            "EnableSubgraph": True,
        },
    )


def build_layer_list(categories):
    """返回 [(category, node_name)], 按 category 分组, 组内按 node_name 排序."""
    m = onnx.load(str(PREPROCESSED), load_external_data=False)
    classes = classify_nodes(m)
    layers = []
    for cat in categories:
        for name in sorted(classes[cat]):
            layers.append((cat, name))
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES,
                        choices=ALL_CATEGORIES,
                        help=f"要逐层测试的类别 (默认中间地带: {DEFAULT_CATEGORIES})")
    parser.add_argument("--frames", type=int, default=None,
                        help="评估帧数 (默认全量 666)")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略已缓存的量化模型, 强制重新量化")
    parser.add_argument("--calib-count", type=int, default=50,
                        help="校准帧数 (leave-one-in 只量化 1 个节点, 50 足够)")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    state_np = load_stats()
    a_q01, a_d = load_action_denorm()
    img_tensors = load_images(args.frames)
    refs = load_refs(len(img_tensors))
    print(f"[eval] {len(img_tensors)} frames", flush=True)

    ensure_preprocessed()
    layers = build_layer_list(args.categories)

    cat_counts = {}
    for cat, _ in layers:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"[plan] {len(layers)} layers: " +
          ", ".join(f"{c}={cat_counts[c]}" for c in args.categories), flush=True)

    results = {}
    t_start = time.time()
    for i, (cat, node) in enumerate(layers):
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(layers) - i - 1) if i > 0 else 0
        print(f"\n[{i+1}/{len(layers)}] ({cat})  elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
        print(f"  node: {node}", flush=True)

        out_path = CACHE_DIR / f"lol_{node.replace('/', '_')}.onnx"
        if not out_path.exists() or args.no_cache:
            t_q = time.time()
            try:
                quantize_one_layer(node, out_path, args.calib_count)
            except Exception as e:
                print(f"  [skip] quantize failed: {e}", flush=True)
                continue
            print(f"  quantized in {time.time()-t_q:.1f}s", flush=True)
        else:
            print(f"  [cache hit]", flush=True)

        try:
            actions, avg_ms = run_inference(out_path, img_tensors, state_np)
        except Exception as e:
            print(f"  [skip] inference failed: {e}", flush=True)
            continue
        tm, md = compute_turn_match(actions, refs, a_q01, a_d)
        drop = 100.0 - tm
        flag = "SAFE" if drop < 0.5 else ("WARN" if drop < 2 else "BAD")
        print(f"  -> turn_match={tm:.2f}%  drop={drop:+.2f}%  "
              f"max_diff={md:.6f}  avg={avg_ms:.1f}ms  [{flag}]", flush=True)
        results[node] = {
            "category": cat,
            "turn_match_pct": round(tm, 4),
            "drop_pct": round(drop, 4),
            "max_diff": md,
            "avg_ms": avg_ms,
        }

    # ===== 汇总 =====
    print(f"\n{'='*100}")
    print("按类别汇总 (leave-one-in, 单层 INT8 敏感度):")
    print(f"{'category':<16} {'count':>6} {'SAFE':>6} {'WARN':>6} {'BAD':>6} "
          f"{'min_drop':>9} {'med_drop':>9} {'max_drop':>9}")
    print(f"{'-'*100}")
    for cat in args.categories:
        vals = [v for v in results.values() if v["category"] == cat]
        if not vals:
            continue
        drops = sorted(v["drop_pct"] for v in vals)
        safe = sum(1 for d in drops if d < 0.5)
        warn = sum(1 for d in drops if 0.5 <= d < 2)
        bad = sum(1 for d in drops if d >= 2)
        med = drops[len(drops) // 2]
        print(f"{cat:<16} {len(vals):>6} {safe:>6} {warn:>6} {bad:>6} "
              f"{drops[0]:>+9.2f} {med:>+9.2f} {drops[-1]:>+9.2f}")

    print(f"\n{'='*100}")
    print(f"坏分子 (drop >= 1.0%, 建议从 INT8 候选中排除):")
    bad_actors = sorted(
        ((name, v) for name, v in results.items() if v["drop_pct"] >= 1.0),
        key=lambda x: -x[1]["drop_pct"],
    )
    if bad_actors:
        for name, v in bad_actors:
            print(f"  [{v['category']:<14}] drop={v['drop_pct']:+.2f}%  "
                  f"max_diff={v['max_diff']:.6f}  {name}")
    else:
        print("  (无, 所有层 drop < 1.0%)")

    print(f"\n{'='*100}")
    print(f"安全层 (drop < 0.5%, 适合 INT8) 共 {sum(1 for v in results.values() if v['drop_pct'] < 0.5)} 个, "
          f"前 10:")
    safe_sorted = sorted(
        ((name, v) for name, v in results.items() if v["drop_pct"] < 0.5),
        key=lambda x: x[1]["drop_pct"],
    )[:10]
    for name, v in safe_sorted:
        print(f"  [{v['category']:<14}] drop={v['drop_pct']:+.2f}%  "
              f"max_diff={v['max_diff']:.6f}  {name}")

    out_path = args.output or (PROJECT_ROOT / "tmp" / "sensitivity_per_layer.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] {len(results)} layers -> {out_path}", flush=True)
    print(f"[total] {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
