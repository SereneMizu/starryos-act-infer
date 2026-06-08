"""
逐层敏感度分析：找出哪些节点可以安全 INT8 量化

对每个可量化节点单独做 INT8 量化，对比输出差异，找出安全的节点。
"""

import json
import sys
import numpy as np
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
import onnx
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FP32 = PROJECT_ROOT / "output" / "train" / "model.onnx"
STATS_PATH = PROJECT_ROOT / "output" / "dataset" / "meta" / "stats.json"
IMG_DIR = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"

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
    return (2 * (np.zeros_like(state_q01) - state_q01) / d - 1).reshape(1, -1).astype(np.float32)


state_np = load_stats()


def get_calibration_data(count=50):
    images = sorted(IMG_DIR.glob("*.jpg"))[:count]
    data = []
    for p in images:
        img = Image.open(p).convert("RGB")
        t = IMAGE_TRANSFORM(img).unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)
        data.append({"images": t, "state": state_np})
    return data


class SimpleCalibrationReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d

    def rewind(self):
        self.idx = 0


def run_inference(model_path, test_data):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    results = []
    for d in test_data:
        out = sess.run(None, d)[0]
        results.append(out[0, 0, :2].copy())
    return np.array(results)


def main():
    calib_data = get_calibration_data(100)
    test_data = get_calibration_data(20)

    from onnxruntime.quantization import shape_inference
    preprocessed = MODEL_FP32.with_suffix(".preprocessed.onnx")
    shape_inference.quant_pre_process(str(MODEL_FP32), str(preprocessed))

    tmp_int8 = PROJECT_ROOT / "output" / "train" / "model_sensitivity_tmp.onnx"

    fp32_results = run_inference(preprocessed, test_data)

    onnx_model = onnx.load(str(preprocessed))
    all_nodes = [n.name for n in onnx_model.graph.node]
    print(f"Total nodes: {len(all_nodes)}")

    reader = SimpleCalibrationReader(calib_data)

    quantizable_nodes = []
    for node in onnx_model.graph.node:
        op = node.op_type
        if op in ("MatMul", "Conv", "Gemm", "Attention", "Mul", "Add"):
            quantizable_nodes.append(node.name)

    print(f"Quantizable nodes: {len(quantizable_nodes)}")

    safe_nodes = []
    sensitive_nodes = []

    for i, node_name in enumerate(quantizable_nodes):
        reader.rewind()
        tmp_int8.unlink(missing_ok=True)

        try:
            quantize_static(
                str(preprocessed),
                str(tmp_int8),
                reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                per_channel=True,
                nodes_to_exclude=[n for n in quantizable_nodes if n != node_name],
            )
        except Exception as e:
            print(f"  [{i+1}/{len(quantizable_nodes)}] {node_name}: SKIP ({e})")
            continue

        try:
            int8_results = run_inference(tmp_int8, test_data)
        except Exception as e:
            print(f"  [{i+1}/{len(quantizable_nodes)}] {node_name}: RUN ERROR ({e})")
            tmp_int8.unlink(missing_ok=True)
            continue

        max_diff = np.abs(fp32_results - int8_results).max()
        mean_diff = np.abs(fp32_results - int8_results).mean()

        direction_flips = 0
        for fp, iq in zip(fp32_results, int8_results):
            fp_turn = "LEFT" if fp[0] < fp[1] else "RIGHT"
            iq_turn = "LEFT" if iq[0] < iq[1] else "RIGHT"
            if fp_turn != iq_turn:
                direction_flips += 1

        label = "SAFE" if direction_flips == 0 and max_diff < 0.01 else "SENSITIVE"
        if direction_flips == 0 and max_diff < 0.01:
            safe_nodes.append(node_name)
        else:
            sensitive_nodes.append(node_name)

        print(f"  [{i+1}/{len(quantizable_nodes)}] {node_name}: max_diff={max_diff:.6f} flips={direction_flips}/{len(test_data)} {label}")
        tmp_int8.unlink(missing_ok=True)

    print(f"\nSafe nodes ({len(safe_nodes)}):")
    for n in safe_nodes:
        print(f"  {n}")
    print(f"\nSensitive nodes ({len(sensitive_nodes)}):")
    for n in sensitive_nodes:
        print(f"  {n}")

    result_path = PROJECT_ROOT / "output" / "sensitivity_analysis.json"
    with open(result_path, "w") as f:
        json.dump({"safe": safe_nodes, "sensitive": sensitive_nodes}, f, indent=2)
    print(f"\nSaved to {result_path}")

    preprocessed.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
