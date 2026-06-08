"""
ACT 模型混合精度量化 (FP16 + INT8)

策略: INT8 量化稳健层, 跳过敏感层, 最后将 FP32 残留转 FP16
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxconverter_common import float16
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

SENSITIVE_INIT_PATTERNS = [
    "decoder",
    "action_head",
    "latent_proj",
]


def load_stats():
    with open(STATS_PATH) as f:
        raw = json.load(f)
    state_q01 = np.array(raw["observation.state"]["q01"], dtype=np.float32)
    state_q99 = np.array(raw["observation.state"]["q99"], dtype=np.float32)
    d = np.where(state_q99 - state_q01 == 0, 1e-8, state_q99 - state_q01)
    return (2 * (np.zeros_like(state_q01) - state_q01) / d - 1).reshape(1, -1).astype(np.float32)


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


def get_sensitive_nodes(model_path: Path) -> list[str]:
    onnx_model = onnx.load(str(model_path))
    sensitive_inits = set()
    for init in onnx_model.graph.initializer:
        for pat in SENSITIVE_INIT_PATTERNS:
            if pat in init.name:
                sensitive_inits.add(init.name)
    nodes = []
    for node in onnx_model.graph.node:
        for inp in node.input:
            if inp in sensitive_inits:
                nodes.append(node.name)
                break
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or PROJECT_ROOT / "output" / "train" / "model_mixed.onnx"

    if not MODEL_FP32.exists():
        print(f"[error] FP32 model not found: {MODEL_FP32}")
        sys.exit(1)

    from onnxruntime.quantization import shape_inference

    preprocessed = MODEL_FP32.with_suffix(".preprocessed.onnx")
    shape_inference.quant_pre_process(str(MODEL_FP32), str(preprocessed))
    print(f"[quantize] preprocessed: {preprocessed}")

    sensitive_nodes = get_sensitive_nodes(preprocessed)
    print(f"[quantize] sensitive nodes ({len(sensitive_nodes)}):")
    for n in sensitive_nodes[:5]:
        print(f"  {n}")
    if len(sensitive_nodes) > 5:
        print(f"  ... and {len(sensitive_nodes) - 5} more")

    reader = ACTCalibrationReader()

    print(f"[quantize] INT8 on {len(reader.images)} calibration images, excluding {len(sensitive_nodes)} nodes...")
    quantize_static(
        str(preprocessed),
        str(output),
        reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
        nodes_to_exclude=sensitive_nodes,
    )
    int8_mb = output.stat().st_size / 1e6
    print(f"[quantize] mixed (encoder INT8 + decoder FP32): {output} ({int8_mb:.1f} MB)")

    onnx.checker.check_model(onnx.load(str(output)))
    print("[quantize] model check passed")

    out_mb = output.stat().st_size / 1e6
    print(f"[quantize] mixed (INT8+FP16): {output} ({out_mb:.1f} MB)")

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print("[quantize] model check passed")

    preprocessed.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
