"""按类别 INT8 敏度分析 (666 帧, CUDA 推理)

对每个算子类别 (Conv / QKV / AttnOut / FFN1 / FFN2) 单独做 INT8 静态量化
(其他全部 FP32), 用 CUDA 跑全量 666 帧评估. 这样能隔离每类算子对 INT8 的
固有敏感度, 指导最终的"哪些类 INT8 / 哪些类 FP16"决策.

## 重要

必须用全量 666 帧评估. 早期版本用 100 帧时, 边界帧刚好没被采样到, 导致
conv-only / qkv-only 出现 "turn drop = 0%" 的假象 (实测 666 帧下 conv-only
掉 1.35%, qkv-only 掉 0.30%).

## 标识方式

结果 JSON 以 preset 名为 key, value 里包含完整 node 列表 (量化后图中的全名).
后续生成混合精度模型时, 可直接取 node 列表喂给 nodes_to_quantize, 无需重新
分类.

## leave-one-in (逐层) 说明

逐层分析能找出类内坏分子, 但: (1) 组合误差 ≠ 单层误差之和, 最终还得跑组合
验证; (2) 部署复杂度高. 对 ACT 模型, 类内敏感度分布相对均匀, 按类别决策已
足够. 如确需逐层, 单独写脚本复用 preprocessed 缓存即可.

用法:
    source scripts/cuda-env.sh
    .venv/bin/python scripts/sensitivity_analysis.py                   # 全部 5 类
    .venv/bin/python scripts/sensitivity_analysis.py --categories conv # 只跑指定类
    .venv/bin/python scripts/sensitivity_analysis.py --frames 100      # 快速测试
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quantize_mixed_onnx import (
    MODEL_FP32, STATS_PATH, IMG_DIR, REF_PATH, IMAGE_TRANSFORM, load_stats,
    classify_nodes, make_calibration_reader, PRESETS,
)

CACHE_DIR = PROJECT_ROOT / "tmp" / "sensitivity_cache"
PREPROCESSED = CACHE_DIR / "preprocessed.onnx"

# preset -> 显示名 + 量化的类别
SENSITIVITY_PRESETS = ["conv-only", "qkv-only", "attn-out-only", "ffn1-only", "ffn2-only"]
PRESET_LABEL = {
    "conv-only": "Conv",
    "qkv-only": "QKV",
    "attn-out-only": "AttnOut",
    "ffn1-only": "FFN1",
    "ffn2-only": "FFN2",
}


def load_action_denorm():
    with open(STATS_PATH) as f:
        raw = json.load(f)
    q01 = np.array(raw["action"]["q01"], dtype=np.float32)
    q99 = np.array(raw["action"]["q99"], dtype=np.float32)
    d = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
    return q01, d


def load_images(count=None):
    paths = sorted(IMG_DIR.glob("*.jpg"))
    if count:
        paths = paths[:count]
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        tensors.append(t)
    return tensors


def load_refs(n):
    with open(REF_PATH) as f:
        return json.load(f)[:n]


def ensure_preprocessed():
    """缓存 quant_pre_process 输出 (MatMul+Add -> Gemm 融合 + shape inference).
    所有 preset 实验共享, 避免每次重复 ~15s 的预处理."""
    if PREPROCESSED.exists():
        return PREPROCESSED
    from onnxruntime.quantization import shape_inference
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[preprocess] quant_pre_process -> {PREPROCESSED.name} ...", flush=True)
    shape_inference.quant_pre_process(str(MODEL_FP32), str(PREPROCESSED))
    print(f"[preprocess] done", flush=True)
    return PREPROCESSED


def quantize_preset(preset, out_path, calib_count=100):
    """把 preset 对应类别的所有层一起做 INT8 静态量化, 其余保持 FP32.
    返回被量化的 node 全名列表."""
    from onnxruntime.quantization import (
        CalibrationMethod, QuantFormat, QuantType, quantize_static,
    )
    m = onnx.load(str(PREPROCESSED), load_external_data=False)
    cls = classify_nodes(m)
    cfg = PRESETS[preset]
    nodes = []
    for cat in ("conv", "ffn1", "ffn2", "qkv", "attn_out_proj"):
        if cfg[cat]:
            nodes += sorted(cls[cat])

    reader = make_calibration_reader("uniform", count=calib_count, threshold=0.0)
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


def run_inference(model_path, img_tensors, state_np, warmup=3):
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(model_path), sess_opts,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    for _ in range(warmup):
        sess.run(None, {"images": img_tensors[0], "state": state_np})
    actions = []
    t0 = time.time()
    for img in img_tensors:
        out = sess.run(None, {"images": img, "state": state_np})[0]
        actions.append(out[0, 0].copy())
    avg_ms = (time.time() - t0) / len(img_tensors) * 1000
    del sess
    return actions, avg_ms


def compute_turn_match(actions, refs, a_q01, a_d):
    match = 0
    max_diff = 0.0
    for act, ref in zip(actions, refs):
        denorm = (act + 1.0) / 2.0 * a_d + a_q01
        l, r = float(denorm[0]), float(denorm[1])
        turn = "LEFT" if l < r else ("RIGHT" if l > r else "STRAIGHT")
        if turn == ref["turn"]:
            match += 1
        max_diff = max(max_diff, abs(l - ref["left_vel"]), abs(r - ref["right_vel"]))
    n = len(actions)
    return match / n * 100, max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=SENSITIVITY_PRESETS,
                        choices=SENSITIVITY_PRESETS,
                        help="要测试的 preset (默认全部 5 类)")
    parser.add_argument("--frames", type=int, default=None,
                        help="评估帧数 (默认全量 666)")
    parser.add_argument("--no-cache", action="store_true",
                        help="忽略已缓存的量化模型, 强制重新量化")
    parser.add_argument("--calib-count", type=int, default=100,
                        help="校准帧数")
    parser.add_argument("--output", type=Path, default=None,
                        help="结果 JSON 输出路径")
    args = parser.parse_args()

    state_np = load_stats()
    a_q01, a_d = load_action_denorm()
    img_tensors = load_images(args.frames)
    refs = load_refs(len(img_tensors))
    print(f"[eval] {len(img_tensors)} frames", flush=True)

    ensure_preprocessed()

    print(f"[plan] {len(args.categories)} presets: {args.categories}", flush=True)

    results = {}
    t_start = time.time()
    for i, preset in enumerate(args.categories):
        elapsed = time.time() - t_start
        print(f"\n[{i+1}/{len(args.categories)}] {preset} ({PRESET_LABEL[preset]})  "
              f"elapsed={elapsed:.0f}s", flush=True)

        out_path = CACHE_DIR / f"cat_{preset}.onnx"
        if not out_path.exists() or args.no_cache:
            t_q = time.time()
            nodes = quantize_preset(preset, out_path, args.calib_count)
            print(f"  quantized {len(nodes)} nodes in {time.time()-t_q:.1f}s", flush=True)
        else:
            print(f"  [cache hit] {out_path.name}", flush=True)
            # 重新算 nodes 列表用于记录
            m = onnx.load(str(PREPROCESSED), load_external_data=False)
            cls = classify_nodes(m)
            cfg = PRESETS[preset]
            nodes = []
            for cat in ("conv", "ffn1", "ffn2", "qkv", "attn_out_proj"):
                if cfg[cat]:
                    nodes += sorted(cls[cat])

        try:
            actions, avg_ms = run_inference(out_path, img_tensors, state_np)
        except Exception as e:
            print(f"  [skip] inference failed: {e}", flush=True)
            continue
        tm, md = compute_turn_match(actions, refs, a_q01, a_d)
        drop = 100.0 - tm
        flag = "SAFE" if drop < 1 else ("WARN" if drop < 5 else "SENSITIVE")
        print(f"  -> turn_match={tm:.2f}%  drop={drop:+.2f}%  "
              f"max_diff={md:.6f}  avg={avg_ms:.1f}ms  [{flag}]", flush=True)
        results[preset] = {
            "label": PRESET_LABEL[preset],
            "nodes": nodes,
            "node_count": len(nodes),
            "turn_match_pct": round(tm, 4),
            "drop_pct": round(drop, 4),
            "max_diff": md,
            "avg_ms": avg_ms,
        }

    # ===== 汇总 (按 drop 升序: 最不敏感在前) =====
    print(f"\n{'='*90}")
    print("按类别敏感度汇总 (单类 INT8, 其余 FP32, 666 帧):")
    print(f"{'preset':<16} {'nodes':>6} {'turn%':>8} {'drop':>8} {'max_diff':>10} {'avg_ms':>8}  结论")
    print(f"{'-'*90}")
    sorted_res = sorted(results.items(), key=lambda x: x[1]["drop_pct"])
    for preset, v in sorted_res:
        drop = v["drop_pct"]
        if drop < 1:
            tag = "可 INT8"
        elif drop < 5:
            tag = "看取舍"
        else:
            tag = "必须 FP16"
        print(f"{preset:<16} {v['node_count']:>6} {v['turn_match_pct']:>8.2f} "
              f"{drop:>+8.2f} {v['max_diff']:>10.6f} {v['avg_ms']:>8.1f}  {tag}")

    out_path = args.output or (PROJECT_ROOT / "tmp" / "sensitivity_per_class.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[results] -> {out_path}", flush=True)
    print(f"[total] {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
