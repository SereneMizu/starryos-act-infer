"""
RKNN-Toolkit2 编译脚本：ONNX (已 FP16 预转换) → RKNN (RK3588)

用法:
    python scripts/rknn_compile.py                         # 默认 FLOAT (不量化)
    python scripts/rknn_compile.py --quantize INT8 --dataset calibration.txt

说明:
    ONNX 权重已在编译前由 scripts/convert_onnx_fp16.py 转为 FP16，
    因此 --quantize FLOAT（默认）即保留 FP16 权重，不做激活量化。
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONNX_PATH = PROJECT_ROOT / "output" / "train" / "model_fp16.onnx"
WORK_DIR = PROJECT_ROOT / "output" / "rknn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(ONNX_PATH),
                        help="输入 ONNX 模型（默认 model_fp16.onnx）")
    parser.add_argument("--target", default="rk3588",
                        choices=["rk3588", "rk3588s", "rk3576", "rk3562", "rk3568", "rk3566"],
                        help="目标平台")
    parser.add_argument("--quantize", default="FLOAT", choices=["FLOAT", "INT8"],
                        help="量化模式；FLOAT=不量化(默认,权重已是FP16), INT8=需--dataset")
    parser.add_argument("--dataset", default=None,
                        help="INT8 量化所需的 calibration 数据集文件")
    parser.add_argument("--output", default=None,
                        help="输出 .rknn 路径")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ERROR: ONNX not found: {onnx_path}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else (
        WORK_DIR / f"act_model_{args.target.lower()}.rknn"
    )

    try:
        from rknn.api import RKNN
    except ImportError:
        try:
            from rknn.model import RKNN
        except ImportError:
            raise SystemExit(
                "ERROR: rknn-toolkit2 未安装。\n"
                "  参考 https://github.com/airockchip/rknn-toolkit2\n"
                "  pip install rknn-toolkit2"
            )

    print("=" * 60)
    print("RKNN-Toolkit2 ACT Model Compilation")
    print(f"  ONNX:    {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Workdir: {WORK_DIR}")
    print(f"  Target:  {args.target}")
    print(f"  Quant:   {args.quantize}")
    print(f"  Output:  {output_path}")
    print("=" * 60)

    rknn = RKNN(verbose=True)

    ret = rknn.config(target_platform=args.target)
    if ret != 0:
        raise SystemExit(f"rknn.config failed: {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f"rknn.load_onnx failed: {ret}")

    if args.quantize == "INT8":
        if not args.dataset:
            raise SystemExit("ERROR: --quantize INT8 需提供 --dataset")
        do_quant = True
        dataset = args.dataset
    else:
        do_quant = False
        dataset = None

    ret = rknn.build(do_quantization=do_quant, dataset=dataset)
    if ret != 0:
        raise SystemExit(f"rknn.build failed: {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise SystemExit(f"rknn.export_rknn failed: {ret}")

    rknn.release()

    print(f"\n[done] {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
