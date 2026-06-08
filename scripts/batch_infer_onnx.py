import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_stats(stats_dir):
    stats_path = Path(stats_dir) / "meta" / "stats.json"
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "state_q01": np.array(raw["observation.state"]["q01"], dtype=np.float32),
        "state_q99": np.array(raw["observation.state"]["q99"], dtype=np.float32),
        "action_q01": np.array(raw["action"]["q01"], dtype=np.float32),
        "action_q99": np.array(raw["action"]["q99"], dtype=np.float32),
    }


def normalize_state(state, stats):
    q01, q99 = stats["state_q01"], stats["state_q99"]
    d = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
    return (2 * (state - q01) / d - 1).reshape(1, -1).astype(np.float32)


def denormalize_action(action, stats):
    q01, q99 = stats["action_q01"], stats["action_q99"]
    d = np.where(q99 - q01 == 0, 1e-8, q99 - q01)
    return ((action + 1) / 2 * d + q01)


def process_image(path):
    img = Image.open(path).convert("RGB")
    t = IMAGE_TRANSFORM(img)
    return t.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)


def determine_turn(left_vel, right_vel):
    if left_vel < right_vel:
        return "LEFT"
    elif left_vel > right_vel:
        return "RIGHT"
    return "STRAIGHT"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    onnx_path = args.model or PROJECT_ROOT / "output" / "train" / "model.onnx"
    output_path = args.output or PROJECT_ROOT / "output" / "infer_results_onnx.json"
    stats_dir = PROJECT_ROOT / "output" / "dataset"
    img_dir = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"

    stats = load_stats(stats_dir)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"Provider: {session.get_providers()}")

    state_dim = session.get_inputs()[1].shape[1]
    state_np = normalize_state(np.zeros(state_dim, dtype=np.float32), stats)

    images = sorted(img_dir.glob("*.jpg"))
    total = len(images)
    print(f"Found {total} images")

    results = []
    t0 = time.time()

    for i, img_path in enumerate(images):
        img_np = process_image(img_path)
        action = session.run(None, {"images": img_np, "state": state_np})[0]
        first_step = denormalize_action(action[0, 0], stats).tolist()

        left_vel, right_vel = first_step[0], first_step[1]
        turn = determine_turn(left_vel, right_vel)

        results.append({
            "frame": img_path.name,
            "left_vel": round(left_vel, 6),
            "right_vel": round(right_vel, 6),
            "gripper_target": round(first_step[2], 6),
            "turn": turn,
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] {img_path.name} | {turn:>8s} | "
                  f"left={left_vel:+.6f} right={right_vel:+.6f} | "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {total} frames in {elapsed:.1f}s ({elapsed/total:.3f}s/frame)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    left_count = sum(1 for r in results if r["turn"] == "LEFT")
    right_count = sum(1 for r in results if r["turn"] == "RIGHT")
    straight_count = sum(1 for r in results if r["turn"] == "STRAIGHT")
    print(f"Summary: LEFT={left_count} RIGHT={right_count} STRAIGHT={straight_count}")


if __name__ == "__main__":
    main()
