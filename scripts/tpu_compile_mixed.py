"""
TPU-MLIR 混合精度编译：FP32 ONNX → MLIR → INT8+BF16 cvimodel (SG2002/CV181X)

流程:
  1. model_transform.py: FP32 ONNX → Top MLIR
  2. run_calibration.py: 生成 cali_table (校准表)
  3. gen_tpu_qtable.py: 从 enc_full 策略生成 qtable (decoder ffn+attn_out → BF16)
  4. model_deploy.py --quantize INT8 --calibration_table --quantize_table: 混合精度 cvimodel

用法 (在 tpu-mlir 容器内):
  python scripts/tpu_compile_mixed.py
  python scripts/tpu_compile_mixed.py --processor cv181x
  python scripts/tpu_compile_mixed.py --skip-calibration --skip-verify
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
CALI_DIR = WORK_DIR / "calibration_data"
CALI_TABLE = WORK_DIR / "act_cali_table"
QTABLE = WORK_DIR / "act_qtable_enc_full"
NEW_CALI_TABLE = WORK_DIR / "new_cali_table.txt"
GEN_QTABLE_SCRIPT = PROJECT_ROOT / "scripts" / "gen_tpu_qtable.py"

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
    }


def prepare_calibration_dataset(cali_dir: Path, num_samples: int = 100):
    """生成多输入校准数据集: 每个样本一个 npz 文件 (images + state)"""
    cali_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(STATS_PATH)

    state_q01 = stats["state_q01"]
    state_q99 = stats["state_q99"]
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_np = (2 * (np.zeros(len(state_q01), dtype=np.float32) - state_q01) / d - 1).reshape(1, -1)

    images = sorted(IMG_DIR.glob("*.jpg"))
    if num_samples < len(images):
        idxs = np.linspace(0, len(images) - 1, num_samples, dtype=int)
        images = [images[i] for i in idxs]

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")
        t = IMAGE_TRANSFORM(img)
        img_np = t.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        np.savez(cali_dir / f"sample_{i:04d}.npz", images=img_np, state=state_np)

    print(f"[calib] Generated {len(images)} calibration samples in {cali_dir}")
    return cali_dir


def run(cmd, cwd=None, env=None):
    cmd = [str(x) for x in cmd]
    print(f"\n{'='*60}\n$ {' '.join(cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, env=env)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(ONNX_PATH))
    parser.add_argument("--quantize", default="INT8", choices=["INT8", "BF16", "F16", "F32"])
    parser.add_argument("--processor", default="cv181x")
    parser.add_argument("--cali-num", type=int, default=100, help="校准样本数量")
    parser.add_argument("--inference-num", type=int, default=30, help="search_qtable 推理样本数")
    parser.add_argument("--expected-cos", type=float, default=0.99, help="期望混精模型 cos 相似度")
    parser.add_argument("--max-float-layers", type=int, default=20, help="qtable 最大浮点层数")
    parser.add_argument("--fp-type", default="BF16", choices=["auto", "F16", "F32", "BF16"])
    parser.add_argument("--transformer", action="store_true", default=True,
                        help="是 transformer 模型 (默认 True)")
    parser.add_argument("--no-transformer", dest="transformer", action="store_false")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="跳过 search_qtable, 用已有 cali_table + qtable 直接 deploy")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--cali-method", default="MSE",
                        help="校准方法 (MSE/KL/MAX/Percentile9999, 逗号分隔)")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TPU-MLIR Mixed-Precision Compilation (INT8 + BF16)")
    print(f"  ONNX:      {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"  Workdir:   {WORK_DIR}")
    print(f"  Quant:     {args.quantize} (敏感层 {args.fp_type})")
    print(f"  Chip:      {args.processor}")
    print(f"  Expected:  cos={args.expected_cos}, max_float={args.max_float_layers}")
    print("=" * 60)

    stats = load_stats(STATS_PATH)
    state_q01 = stats["state_q01"]
    state_q99 = stats["state_q99"]
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_np = (2 * (np.zeros(len(state_q01), dtype=np.float32) - state_q01) / d - 1).reshape(1, -1)
    MODEL_INPUTS["state"] = list(state_np.shape)

    # Step 1: model_transform (FP32 ONNX → Top MLIR)
    if not MLIR_PATH.exists():
        print(f"\n[1/4] ONNX → Top MLIR (model_transform) ...")
        images_dir = sorted(IMG_DIR.glob("*.jpg"))
        if not images_dir:
            sys.exit("ERROR: no images found in dataset")
        test_img = images_dir[0]
        img = Image.open(test_img).convert("RGB")
        t = IMAGE_TRANSFORM(img)
        img_np = t.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        test_npz = WORK_DIR / "act_test_input.npz"
        np.savez(test_npz, images=img_np, state=state_np)

        input_shapes_str = ",".join(str(MODEL_INPUTS[name]) for name in ["images", "state"])
        run([
            "model_transform.py",
            "--model_name", "act_model",
            "--model_def", str(onnx_path),
            "--input_shapes", f"[{input_shapes_str}]",
            "--test_input", str(test_npz),
            "--test_result", str(WORK_DIR / "act_top_outputs.npz"),
            "--mlir", str(MLIR_PATH),
        ], cwd=WORK_DIR)
    else:
        print(f"\n[1/4] MLIR exists: {MLIR_PATH}, skipping model_transform.")

    if args.quantize != "INT8":
        # 非混合精度: 直接 model_deploy
        output_name = f"act_model_{args.processor}_{args.quantize.lower()}.cvimodel"
        output_path = WORK_DIR / output_name
        print(f"\n[2/4] Skip calibration (quantize={args.quantize})")
        print(f"[3/4] Skip qtable (quantize={args.quantize})")
        print(f"[4/4] MLIR → {args.quantize} cvimodel (model_deploy) ...")
        deploy_cmd = [
            "model_deploy.py",
            "--mlir", str(MLIR_PATH),
            "--quantize", args.quantize,
            "--processor", args.processor,
            "--model", str(output_path),
        ]
        if args.skip_verify:
            deploy_cmd += ["--skip_validation"]
        else:
            test_npz = WORK_DIR / "act_test_input.npz"
            if test_npz.exists():
                deploy_cmd += [
                    "--test_input", str(test_npz),
                    "--test_reference", str(WORK_DIR / "act_top_outputs.npz"),
                ]
            else:
                deploy_cmd += ["--skip_validation"]
        run(deploy_cmd, cwd=WORK_DIR)
        print(f"\nDone: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
        return

    # Step 2: run_calibration (生成 cali_table, 不做 search_qtable)
    if not args.skip_calibration:
        print(f"\n[2/4] Preparing calibration dataset ...")
        prepare_calibration_dataset(CALI_DIR, args.cali_num)

        print(f"\n[3/4] run_calibration (生成 cali_table) ...")
        cali_cmd = [
            "run_calibration.py",
            str(MLIR_PATH),
            "--dataset", str(CALI_DIR),
            "--input_num", str(args.cali_num),
            "--cali_method", args.cali_method.lower(),
            "--chip", args.processor,
            "-o", str(CALI_TABLE),
        ]
        run(cali_cmd, cwd=WORK_DIR)
    else:
        print(f"\n[2/4] Skip calibration (--skip-calibration)")
        if not CALI_TABLE.exists():
            sys.exit(f"ERROR: cali_table not found: {CALI_TABLE}")

    # Step 3: 从 enc_full 策略生成 qtable (复用 quantize_mixed_onnx.py 的分层选择)
    print(f"\n[3/4] Generating qtable from enc_full strategy ...")
    run([sys.executable, str(GEN_QTABLE_SCRIPT), "--mlir", str(MLIR_PATH), "--output", str(QTABLE)],
        cwd=WORK_DIR)

    if not QTABLE.exists():
        sys.exit(f"ERROR: qtable generation failed: {QTABLE}")

    deploy_cali_table = CALI_TABLE

    # Step 4: model_deploy with qtable
    output_name = f"act_model_{args.processor}_mixed.cvimodel"
    output_path = WORK_DIR / output_name

    print(f"\n[4/4] model_deploy --quantize INT8 --quantize_table (混合精度) ...")
    deploy_cmd = [
        "model_deploy.py",
        "--mlir", str(MLIR_PATH),
        "--quantize", "INT8",
        "--processor", args.processor,
        "--calibration_table", str(deploy_cali_table),
        "--quantize_table", str(QTABLE),
        "--model", str(output_path),
    ]

    if args.skip_verify:
        deploy_cmd += ["--skip_validation"]
    else:
        test_npz = WORK_DIR / "act_test_input.npz"
        if test_npz.exists():
            deploy_cmd += [
                "--test_input", str(test_npz),
                "--test_reference", str(WORK_DIR / "act_top_outputs.npz"),
            ]
        else:
            deploy_cmd += ["--skip_validation"]

    run(deploy_cmd, cwd=WORK_DIR)

    print(f"\n{'='*60}")
    print(f"Done: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
    print(f"  cali_table: {deploy_cali_table}")
    print(f"  qtable:     {QTABLE}")

    # Print qtable content
    if QTABLE.exists():
        print(f"\n--- Quantize Table (sensitive layers → {args.fp_type}) ---")
        with open(QTABLE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    print(f"  {line}")

    print(f"\n=== Next Steps ===")
    print(f"  Copy {output_path} to SG2002 device")


if __name__ == "__main__":
    main()