"""
TPU-MLIR 编译脚本：ONNX → MLIR → F16 cvimodel (SG2002/CV186X)

前置条件：
    pip install tpu_mlir   (在 sophgo/tpuc_dev Docker 内)

用法:
    python scripts/tpu_compile.py                # 默认 F16
    python scripts/tpu_compile.py --quantize BF16
    python scripts/tpu_compile.py --quantize INT8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONNX_PATH = PROJECT_ROOT / "output" / "train" / "model.onnx"
STATS_PATH = PROJECT_ROOT / "output" / "dataset" / "meta" / "stats.json"
IMG_DIR = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"
WORK_DIR = PROJECT_ROOT / "output" / "tpu"
MLIR_PATH = WORK_DIR / "act_model.mlir"
OUTPUT_NAME = "act_model_cv186x_f16.cvimodel"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODEL_INPUTS = {
    "images": [1, 1, 3, 224, 224],
    "state":  None,
}


def load_stats(stats_path):
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "state_q01": np.array(raw["observation.state"]["q01"], dtype=np.float32),
        "state_q99": np.array(raw["observation.state"]["q99"], dtype=np.float32),
        "action_q01": np.array(raw["action"]["q01"], dtype=np.float32),
        "action_q99": np.array(raw["action"]["q99"], dtype=np.float32),
    }


def prepare_test_input(test_img_path, stats):
    img = Image.open(test_img_path).convert("RGB")
    t = IMAGE_TRANSFORM(img)
    img_np = t.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)

    state_q01 = stats["state_q01"]
    state_q99 = stats["state_q99"]
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_np = (2 * (np.zeros(len(state_q01), dtype=np.float32) - state_q01) / d - 1).reshape(1, -1)

    MODEL_INPUTS["state"] = list(state_np.shape)

    test_npz = WORK_DIR / "act_test_input.npz"
    np.savez(test_npz, images=img_np, state=state_np)
    return test_npz


def run(args, **kwargs):
    cmd = [str(x) for x in args]
    print(f"\n{'='*60}\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    return subprocess.run(cmd, check=True, cwd=str(WORK_DIR), **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", default="BF16", choices=["F16", "BF16", "INT8", "F32"])
    parser.add_argument("--processor", default="cv181x")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true", help="Skip cmodel accuracy verification")
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TPU-MLIR ACT Model Compilation")
    print(f"  ONNX:    {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)" if ONNX_PATH.exists() else "  ONNX: NOT FOUND")
    print(f"  Workdir: {WORK_DIR}")
    print(f"  Quant:   {args.quantize}")
    print(f"  Chip:    {args.processor}")
    print("=" * 60)

    if not ONNX_PATH.exists():
        print("\n[1/4] Exporting ONNX model ...")
        run([sys.executable, str(PROJECT_ROOT / "scripts" / "export_onnx.py")])
    else:
        print("\n[1/4] ONNX model found, skipping export.")

    stats = load_stats(STATS_PATH)

    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        sys.exit("ERROR: no images found in dataset")
    test_img = images[0]
    print(f"[2/4] Preparing test input from {test_img.name} ...")
    test_npz = prepare_test_input(test_img, stats)

    input_shapes_str = ",".join(
        str(MODEL_INPUTS[name]) for name in ["images", "state"]
    )
    input_shapes_arg = f"[{input_shapes_str}]"

    mlir_exists = MLIR_PATH.exists()
    if not mlir_exists:
        print(f"[2/4] ONNX → MLIR (model_transform) ...")
        run([
            "model_transform.py",
            "--model_name", "act_model",
            "--model_def", str(ONNX_PATH),
            "--input_shapes", input_shapes_arg,
            "--test_input", str(test_npz),
            "--test_result", str(WORK_DIR / "act_top_outputs.npz"),
            "--mlir", str(MLIR_PATH),
        ])
    else:
        print(f"[2/4] MLIR exists, skipping model_transform.")

    output_name = f"act_model_{args.processor}_{args.quantize.lower()}.cvimodel"
    output_path = WORK_DIR / output_name

    print(f"[3/4] MLIR → {args.quantize} cvimodel (model_deploy) ...")
    deploy_cmd = [
        "model_deploy.py",
        "--mlir", str(MLIR_PATH),
        "--quantize", args.quantize,
        "--processor", args.processor,
        "--num_core", "1",
        "--addr_mode", "io_alone",
        "--model", str(output_path),
        "--skip_validation",
    ]

    if test_npz.exists() and not args.skip_verify:
        deploy_cmd += [
            "--test_input", str(test_npz),
            "--test_reference", str(WORK_DIR / "act_top_outputs.npz"),
        ]

    run(deploy_cmd)

    print(f"\n[4/4] Done: {output_path}  ({output_path.stat().st_size/1e6:.1f} MB)")

    model_tool = os.environ.get("MODEL_TOOL", "model_tool")
    if os.path.exists(model_tool) or True:
        print("\n--- Model Info ---")
        try:
            run([model_tool, "--info", str(output_path)])
        except Exception:
            pass

    print("\n=== Next Steps ===")
    print(f"  Copy {output_path} to SG2002 device")
    print(f"  Use cvitek_tpu_sdk CVI_NN_* API to load & run")


if __name__ == "__main__":
    main()
