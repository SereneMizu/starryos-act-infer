"""
ONNX FP32 → FP16 转换（权重层面，保留输入/输出为 FP32）
输出给 rknn-toolkit2 编译，避免其 calibration 流程。
"""

import argparse
from pathlib import Path

import onnx
from onnxconverter_common import float16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="output/train/model.onnx")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--keep-io-types", action="store_true", default=True,
                        help="保持输入/输出层为 FP32（默认开启）")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"ERROR: {src} not found")

    dst = Path(args.output) if args.output else src.with_name(f"{src.stem}_fp16.onnx")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fp16] loading {src} ({src.stat().st_size / 1e6:.1f} MB) ...")
    model = onnx.load(str(src))

    print("[fp16] converting weights FP32 → FP16 ...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=args.keep_io_types,
    )

    onnx.save(model_fp16, str(dst))
    print(f"[fp16] done → {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
