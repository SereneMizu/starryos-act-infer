"""
ACT 模型 BF16 量化 — 全图 BF16（含 IO）

ONNX Runtime CPU 会用软件模拟 BF16 计算。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FP32 = PROJECT_ROOT / "output" / "train" / "model.onnx"


def convert_initializer_to_bf16(init: TensorProto) -> TensorProto:
    arr = numpy_helper.to_array(init)
    bf16_arr = arr.astype(np.float32).view(np.uint32) >> 16
    bf16_arr = bf16_arr.astype(np.uint16)

    new_init = TensorProto()
    new_init.name = init.name
    new_init.data_type = TensorProto.BFLOAT16
    new_init.dims.extend(init.dims)
    new_init.raw_data = bf16_arr.tobytes()
    return new_init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or PROJECT_ROOT / "output" / "train" / "model_bf16.onnx"

    if not MODEL_FP32.exists():
        print(f"[error] FP32 model not found: {MODEL_FP32}")
        sys.exit(1)

    fp32_mb = MODEL_FP32.stat().st_size / 1e6
    print(f"[bf16] FP32 model: {fp32_mb:.1f} MB")

    model = onnx.load(str(MODEL_FP32))

    converted = 0
    for i, init in enumerate(model.graph.initializer):
        if init.data_type == TensorProto.FLOAT:
            model.graph.initializer[i].CopyFrom(convert_initializer_to_bf16(init))
            converted += 1
    print(f"[bf16] converted {converted} initializers to BF16")

    for vi in model.graph.value_info:
        if vi.type.HasField('tensor_type') and vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.BFLOAT16

    for inp in model.graph.input:
        if inp.type.HasField('tensor_type') and inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.BFLOAT16

    for out in model.graph.output:
        if out.type.HasField('tensor_type') and out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.BFLOAT16

    onnx.save(model, str(output))
    bf16_mb = output.stat().st_size / 1e6
    print(f"[bf16] output: {output} ({bf16_mb:.1f} MB)")

    onnx.checker.check_model(onnx.load(str(output)))
    print("[bf16] model check passed")


if __name__ == "__main__":
    main()
