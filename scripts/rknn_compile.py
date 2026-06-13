"""
RKNN-Toolkit2 编译脚本：ONNX → RKNN (RK3588)

前置条件：
    pip install rknn-toolkit2   (在主机 .venv 内，x86_64)
    参考: https://github.com/airockchip/rknn-toolkit2

用法:
    python scripts/rknn_compile.py                         # 默认不量化 (浮点, NPU 以 fp16 执行)
    python scripts/rknn_compile.py --target rk3588
    python scripts/rknn_compile.py --quantize INT8 --dataset calibration.txt

说明:
    RK3588 NPU 硬件不支持真 fp32 计算；浮点模型在 NPU 上以 fp16 执行。
    因此 do_quantization=False（默认）即为"不量化"能达到的最高精度，
    正确性最好，对应赛题"用最大的 onnx 模型"的诉求。
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ONNX_PATH = PROJECT_ROOT / "output" / "train" / "model.onnx"
WORK_DIR = PROJECT_ROOT / "output" / "rknn"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=str(ONNX_PATH),
                        help="输入 ONNX 模型（默认最大的 model.onnx）")
    parser.add_argument("--target", default="rk3588",
                        choices=["rk3588", "rk3588s", "rk3576", "rk3562", "rk3568", "rk3566"],
                        help="目标平台")
    parser.add_argument("--quantize", default=None, choices=[None, "INT8", "FP16"],
                        help="量化模式；None=不量化(浮点,默认)")
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
            from rknn.model import RKNN  # toolkit2 v2
        except ImportError:
            raise SystemExit(
                "ERROR: rknn-toolkit2 未安装。\n"
                "  参考 https://github.com/airockchip/rknn-toolkit2\n"
                "  pip install rknn-toolkit2  (注意: 需匹配 Python 版本的 wheel)"
            )

    print("=" * 60)
    print("RKNN-Toolkit2 ACT Model Compilation")
    print(f"  ONNX:    {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Workdir: {WORK_DIR}")
    print(f"  Target:  {args.target}")
    print(f"  Quant:   {args.quantize or 'NONE (float, NPU runs fp16)'}")
    print(f"  Output:  {output_path}")
    print("=" * 60)

    rknn = RKNN(verbose=True)

    # 模型输入已由推理程序完成 ImageNet 归一化，且以 float32 喂入；
    # rknn 的 mean/std 归一化只作用于 uint8 图像输入，对 float32 无效。
    # 省略 mean_values/std_values，rknn 自动设 mean=0/std=1（identity）。
    ret = rknn.config(target_platform=args.target)
    if ret != 0:
        raise SystemExit(f"rknn.config failed: {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise SystemExit(f"rknn.load_onnx failed: {ret}")

    do_quant = args.quantize is not None
    dataset = args.dataset if do_quant else None
    if do_quant and args.quantize == "INT8" and not dataset:
        print("[warn] INT8 量化未提供 --dataset，将退化为非量化浮点模型")
        do_quant = False

    ret = rknn.build(do_quantization=do_quant, dataset=dataset)
    if ret != 0:
        raise SystemExit(f"rknn.build failed: {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise SystemExit(f"rknn.export_rknn failed: {ret}")

    rknn.release()

    print(f"\n[done] {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    print("  拷贝到 RK3588 设备，用 librknnrt.so 加载运行 (rknn_init/inputs_set/run/outputs_get)")


if __name__ == "__main__":
    main()
