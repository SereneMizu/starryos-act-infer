"""
ACT 模型 ONNX 导出

用法:
    .venv\\Scripts\\python export_onnx.py
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

MODEL_PATH = PROJECT_ROOT / "output" / "train" / "model.pt"
ONNX_PATH = PROJECT_ROOT / "output" / "train" / "model.onnx"
ONNX_OPSET = 18


class ACTInferenceWrapper(nn.Module):
    """
    精简 wrapper: 直接调用 model.forward(infer_cvae=False).

    infer_cvae=False 让 model 内部走 latent=torch.zeros() 分支 (= E[latent]),
    与 batch_infer_torch.py 完全同代码路径, 保证 ONNX 部署与 PyTorch 推理数值一致.
    无需重写 forward 逻辑.
    """

    def __init__(self, model: ACTModel):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.model.forward(images, state, action_target=None, infer_cvae=False)["action"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config_dict = ckpt.get("config", {})
    else:
        state_dict, config_dict = ckpt, {}

    config = ACTConfig(**config_dict)
    model = ACTModel(config)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    wrapper = ACTInferenceWrapper(model).to(device).eval()

    state_dim = config_dict.get("state_dim", 2)
    dummy_images = torch.randn(1, 1, 3, 224, 224, device=device)
    dummy_state = torch.randn(1, state_dim, device=device)

    wrapper.eval()
    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_state),
        str(ONNX_PATH),
        opset_version=ONNX_OPSET,
        input_names=["images", "state"],
        output_names=["action"],
        do_constant_folding=True,
        dynamo=False,
        external_data=False,
    )

    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"[onnx] exported: {ONNX_PATH}  ({size_mb:.1f} MB)")

    import onnx
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model, full_check=True)
    print("[onnx] model check passed")


if __name__ == "__main__":
    main()
