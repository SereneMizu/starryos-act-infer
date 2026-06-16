"""
ACT 模型逐类量化敏感度分析

对每个 preset (只量化某一类层) 生成纯 INT8 模型 (不做 FP16 转换, 避免 FP16 噪声),
然后用 ORT Python API 跑评估集, 统计:
  - turn match 与 FP32 baseline 的差异
  - left/right vel 的 MSE
  - 逐帧 turn 差异

这样能精确知道每一类层对量化的敏感度, 指导最终混合量化策略.

用法:
    .venv/bin/python scripts/sensitivity_analysis.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from quantize_mixed_onnx import (
    MODEL_FP32, STATS_PATH, IMG_DIR, IMAGE_TRANSFORM, load_stats,
    stage1_int8_quantize, PRESETS,
)

EVAL_COUNT = 100  # 评估帧数 (足够看出 turn match 趋势, 且省时省内存)
SENSITIVITY_PRESETS = ["conv-only", "ffn1-only", "ffn2-only", "qkv-only", "attn-out-only"]


def load_eval_data():
    """返回 (images, states, refs) 用于评估"""
    with open(STATS_PATH) as f:
        raw = json.load(f)
    state_q01 = np.array(raw["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.array(raw["observation.state"]["q99"], dtype=np.float32)
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_normed = (2 * (np.zeros_like(state_q01) - state_q01) / d - 1).reshape(1, -1).astype(np.float32)

    # 加载 ref
    ref_path = PROJECT_ROOT / "output" / "infer_results_onnx.json"
    with open(ref_path) as f:
        refs_all = json.load(f)
    refs = refs_all[:EVAL_COUNT]

    images = sorted(IMG_DIR.glob("*.jpg"))[:EVAL_COUNT]
    img_tensors = []
    for p in images:
        img = Image.open(p).convert("RGB")
        t = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        img_tensors.append(t)

    return img_tensors, state_normed, refs


def run_inference(model_path, img_tensors, state_np):
    """加载模型并推理, 返回 (actions_list, avg_ms)"""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])

    actions = []
    import time
    t0 = time.time()
    for img in img_tensors:
        out = sess.run(None, {"images": img, "state": state_np})[0]
        # out shape: [1, chunk_size, action_dim], 取 [0, 0]
        actions.append(out[0, 0])  # [action_dim]
    elapsed = time.time() - t0
    avg_ms = elapsed / len(img_tensors) * 1000
    return actions, avg_ms


def compute_metrics(actions, refs):
    """返回 turn_match%, max_left_diff, max_right_diff, mse"""
    action_q01 = None
    with open(STATS_PATH) as f:
        raw = json.load(f)
    action_q01 = np.array(raw["action"]["q01"], dtype=np.float32)
    action_q99 = np.array(raw["action"]["q99"], dtype=np.float32)
    d = np.where(action_q99 - action_q01 == 0, 1e-8, action_q99 - action_q01)

    turn_match = 0
    max_left_diff = 0.0
    max_right_diff = 0.0
    se_sum = 0.0

    for act, ref in zip(actions, refs):
        # denormalize
        denorm = (act + 1.0) / 2.0 * d + action_q01
        left = float(denorm[0])
        right = float(denorm[1])
        turn = "LEFT" if left < right else ("RIGHT" if left > right else "STRAIGHT")

        ref_left = ref["left_vel"]
        ref_right = ref["right_vel"]
        ref_turn = ref["turn"]

        if turn == ref_turn:
            turn_match += 1
        max_left_diff = max(max_left_diff, abs(left - ref_left))
        max_right_diff = max(max_right_diff, abs(right - ref_right))
        se_sum += (left - ref_left) ** 2 + (right - ref_right) ** 2

    n = len(actions)
    return {
        "turn_match_pct": turn_match / n * 100,
        "max_left_diff": max_left_diff,
        "max_right_diff": max_right_diff,
        "mse": se_sum / n,
    }


def main():
    print(f"[sensitivity] loading {EVAL_COUNT} eval frames ...")
    img_tensors, state_np, refs = load_eval_data()
    print(f"[sensitivity] loaded {len(img_tensors)} frames")

    tmpdir = Path(tempfile.mkdtemp(prefix="act_sens_"))
    print(f"[sensitivity] tmp dir: {tmpdir}")

    # 1. FP32 baseline
    print(f"\n=== FP32 baseline ===")
    base_actions, base_ms = run_inference(MODEL_FP32, img_tensors, state_np)
    base_metrics = compute_metrics(base_actions, refs)
    print(f"  avg={base_ms:.1f}ms  turn_match={base_metrics['turn_match_pct']:.1f}%  "
          f"max_L={base_metrics['max_left_diff']:.6f}  max_R={base_metrics['max_right_diff']:.6f}")

    results = {"FP32": {"avg_ms": base_ms, **base_metrics}}

    # 2. 逐类量化
    for preset in SENSITIVITY_PRESETS:
        print(f"\n=== {preset} (only this category quantized to INT8) ===")
        out_path = tmpdir / f"{preset}.onnx"
        try:
            stage1_int8_quantize(MODEL_FP32, out_path, preset, calibrate_method="MinMax")
        except Exception as e:
            print(f"  [skip] quantize failed: {e}")
            continue

        actions, avg_ms = run_inference(out_path, img_tensors, state_np)
        metrics = compute_metrics(actions, refs)
        # 精度损失 = 相对 baseline 的 turn_match 下降 + diff 增加
        turn_drop = base_metrics["turn_match_pct"] - metrics["turn_match_pct"]
        print(f"  avg={avg_ms:.1f}ms  turn_match={metrics['turn_match_pct']:.1f}%  "
              f"(drop {turn_drop:+.1f})  "
              f"max_L={metrics['max_left_diff']:.6f}  max_R={metrics['max_right_diff']:.6f}")
        results[preset] = {"avg_ms": avg_ms, "turn_drop": turn_drop, **metrics}

    # 3. 汇总
    print(f"\n{'='*80}")
    print(f"{'Preset':<20} {'avg_ms':>8} {'turn%':>7} {'drop':>7} {'max_L':>10} {'max_R':>10} {'mse':>10}")
    print(f"{'-'*80}")
    for k, v in results.items():
        drop = v.get("turn_drop", 0)
        print(f"{k:<20} {v['avg_ms']:>8.1f} {v['turn_match_pct']:>7.1f} {drop:>+7.1f} "
              f"{v['max_left_diff']:>10.6f} {v['max_right_diff']:>10.6f} {v['mse']:>10.6f}")

    # 敏感度排序 (turn_drop 越大越敏感)
    print(f"\n{'='*80}")
    print("敏感度排序 (turn_drop 越大 = 越敏感 = 越不该量化):")
    sensitivity = [(k, v.get("turn_drop", 0)) for k, v in results.items() if k != "FP32"]
    sensitivity.sort(key=lambda x: -x[1])
    for k, drop in sensitivity:
        print(f"  {k:<20} turn_drop={drop:+.1f}")

    print(f"\n[sensitivity] tmp models in {tmpdir} (可手动清理)")


if __name__ == "__main__":
    main()
