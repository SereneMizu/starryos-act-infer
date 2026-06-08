"""
ACT 模型量化: FP16 / INT8 静态量化

用法:
    .venv/bin/python scripts/quantize_onnx.py            # FP16
    .venv/bin/python scripts/quantize_onnx.py --int8     # INT8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FP32 = PROJECT_ROOT / "output" / "train" / "model.onnx"
STATS_PATH = PROJECT_ROOT / "output" / "dataset" / "meta" / "stats.json"
IMG_DIR = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"
CALIB_COUNT = 200

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_stats():
    with open(STATS_PATH) as f:
        raw = json.load(f)
    state_q01 = np.array(raw["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.array(raw["observation.state"]["q99"], dtype=np.float32)
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    state_normed = (2 * (np.zeros_like(state_q01) - state_q01) / d - 1).reshape(1, -1).astype(np.float32)
    return state_normed


class ACTCalibrationReader(CalibrationDataReader):
    def __init__(self):
        self.state_np = load_stats()
        self.images = sorted(IMG_DIR.glob("*.jpg"))[:CALIB_COUNT]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.images):
            return None
        img = Image.open(self.images[self.idx]).convert("RGB")
        tensor = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        self.idx += 1
        return {"images": tensor, "state": self.state_np}

    def rewind(self):
        self.idx = 0


def quantize_fp32_to_int8(model_fp32: Path, model_int8: Path):
    from onnxruntime.quantization import shape_inference

    preprocessed = model_fp32.with_suffix(".preprocessed.onnx")
    shape_inference.quant_pre_process(str(model_fp32), str(preprocessed))

    reader = ACTCalibrationReader()
    print(f"[quantize] INT8 calibration on {len(reader.images)} images...")

    quantize_static(
        str(preprocessed),
        str(model_int8),
        reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
    )

    preprocessed.unlink(missing_ok=True)
    return model_int8


def quantize_fp32_to_fp16(model_fp32: Path, model_fp16: Path):
    from onnxconverter_common import float16

    onnx_model = onnx.load(str(model_fp32))
    onnx_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save(onnx_model, str(model_fp16))
    return model_fp16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8", action="store_true", help="Use INT8 instead of FP16")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not MODEL_FP32.exists():
        print(f"[error] FP32 model not found: {MODEL_FP32}")
        sys.exit(1)

    print(f"[quantize] FP32 model: {MODEL_FP32} ({MODEL_FP32.stat().st_size / 1e6:.1f} MB)")

    if args.int8:
        output = args.output or PROJECT_ROOT / "output" / "train" / "model_int8.onnx"
        quantize_fp32_to_int8(MODEL_FP32, output)
    else:
        output = args.output or PROJECT_ROOT / "output" / "train" / "model_fp16.onnx"
        quantize_fp32_to_fp16(MODEL_FP32, output)

    out_mb = output.stat().st_size / 1e6
    print(f"[quantize] output: {output} ({out_mb:.1f} MB)")

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print("[quantize] model check passed")


if __name__ == "__main__":
    main()
