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
    将 ACTModel 包装为纯确定性模型，消除 CVAE 随机采样。

    推理策略: 固定零 latent 向量（等价于先验均值），跳过随机采样。
    """

    def __init__(self, model: ACTModel):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(self, images: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch_size = images.shape[0]

        with torch.no_grad():
            latent = torch.zeros(batch_size, config.latent_dim, device=images.device, dtype=images.dtype)

            vision_features = self.model.vision_encoder(images)
            state_features = self.model.state_encoder(state)
            latent_features = self.model.latent_proj(latent).unsqueeze(1)

            encoder_in = torch.cat([latent_features, state_features, vision_features], dim=1)

            seq_len = encoder_in.shape[1]
            if seq_len <= self.model.encoder_pos_embed.num_embeddings:
                pos_embed = self.model.encoder_pos_embed.weight[:seq_len].unsqueeze(0)
            else:
                repeat_count = (seq_len // self.model.encoder_pos_embed.num_embeddings) + 1
                pos_embed = self.model.encoder_pos_embed.weight.repeat(1, repeat_count, 1)[:, :seq_len]

            encoder_out = self.model.encoder(encoder_in, pos_embed=pos_embed)

            decoder_pos_embed = self.model.decoder_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            decoder_in = torch.zeros(
                batch_size, config.action_chunk_size, config.hidden_dim,
                device=images.device, dtype=images.dtype,
            ) + decoder_pos_embed

            decoder_out = self.model.decoder(
                decoder_in, encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=pos_embed,
            )

            action_pred = self.model.action_head(decoder_out)

        return action_pred


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
        dynamo=True,
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
