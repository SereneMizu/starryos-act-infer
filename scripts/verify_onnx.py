"""
ONNX 推理验证: PyTorch vs ONNX 多次运行对比，检查可复现性

用法:
    .venv\\Scripts\\python verify_onnx.py [--runs N]
"""

import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

MODEL_PATH = PROJECT_ROOT / "output" / "train" / "model.pt"
ONNX_PATH = PROJECT_ROOT / "output" / "train" / "model.onnx"
STATS_PATH = PROJECT_ROOT / "output" / "dataset" / "meta" / "stats.json"

IMAGE_PATHS = [
    PROJECT_ROOT / "output/dataset/videos/observation.images.fpv/chunk-000/frame_000000.jpg",
    PROJECT_ROOT / "output/dataset/videos/observation.images.fpv/chunk-000/frame_000227.jpg",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ACTInferenceWrapper(nn.Module):
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


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config_dict = ckpt.get("config", {})
    else:
        state_dict, config_dict = ckpt, {}
    config = ACTConfig(**config_dict)
    model = ACTModel(config)
    model.load_state_dict(state_dict)
    return model.to(device).eval(), config_dict


def load_stats(stats_path: Path) -> dict:
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "state_q01": raw["observation.state"]["q01"],
        "state_q99": raw["observation.state"]["q99"],
        "action_q01": raw["action"]["q01"],
        "action_q99": raw["action"]["q99"],
    }


def process_image(path: Path, device: torch.device) -> torch.Tensor:
    img: Image.Image = Image.open(path).convert("RGB")
    t: torch.Tensor = IMAGE_TRANSFORM(img)
    return t.unsqueeze(0).unsqueeze(0).to(device)


def normalize_state(state: list, stats: dict, device: torch.device) -> torch.Tensor:
    q01 = torch.tensor(stats["state_q01"], dtype=torch.float32, device=device)
    q99 = torch.tensor(stats["state_q99"], dtype=torch.float32, device=device)
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8, device=device), d)
    return 2 * (s - q01) / d - 1


def denormalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    q01 = torch.tensor(stats["action_q01"], dtype=torch.float32)
    q99 = torch.tensor(stats["action_q99"], dtype=torch.float32)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8), d)
    return (action.cpu() + 1) / 2 * d + q01


def run_once(wrapper, ort_session, image_tensors, state_tensor, stats, device):
    """单次运行: 对所有测试图片执行 PyTorch 和 ONNX 推理，返回原始 action 输出。"""
    pt_results = {}
    ort_results = {}

    for img_path, img_tensor in zip(IMAGE_PATHS, image_tensors):
        name = img_path.name

        wrapper.eval()
        with torch.no_grad():
            pt_action = wrapper(img_tensor, state_tensor)

        ort_output = ort_session.run(None, {
            "images": img_tensor.cpu().numpy(),
            "state": state_tensor.cpu().numpy(),
        })[0]

        pt_results[name] = pt_action.cpu()
        ort_results[name] = torch.from_numpy(ort_output.copy())

    return pt_results, ort_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5, help="重复运行次数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}" + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    model, config_dict = load_model(MODEL_PATH, device)
    wrapper = ACTInferenceWrapper(model).to(device).eval()
    stats = load_stats(STATS_PATH)

    state_dim = config_dict.get("state_dim", 2)
    state_tensor = normalize_state([0.0] * state_dim, stats, device)

    image_tensors = [process_image(p, device) for p in IMAGE_PATHS]

    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(str(ONNX_PATH), sess_options, providers=["CPUExecutionProvider"])
    print(f"[ort] provider: {ort_session.get_providers()}")

    dim_labels = ["left_vel", "right_vel", "gripper_target"]

    all_runs = []
    for run_idx in range(args.runs):
        pt_results, ort_results = run_once(wrapper, ort_session, image_tensors, state_tensor, stats, device)
        all_runs.append((pt_results, ort_results))

    # ---- 检查 1: 每次运行 PyTorch vs ONNX 差异 ----
    print("\n" + "=" * 76)
    print("  [1] PyTorch vs ONNX 逐次对比")
    print("=" * 76)

    for run_idx, (pt_results, ort_results) in enumerate(all_runs):
        print(f"\n  run {run_idx}:")
        for name in [p.name for p in IMAGE_PATHS]:
            pt = pt_results[name]
            ort = ort_results[name]
            diff = torch.abs(pt - ort).max().item()
            denorm_pt = denormalize_action(pt[0, 0], stats)
            vals = denorm_pt.cpu().tolist()
            vals_str = "  ".join(f"{dim_labels[i]}={v:+.6f}" for i, v in enumerate(vals[:len(dim_labels)]))
            print(f"    {name:<20} max_diff={diff:.2e}  |  {vals_str}")

    # ---- 检查 2: 多次运行之间的可复现性 ----
    print("\n" + "=" * 76)
    print("  [2] 多次运行可复现性检查 (跨 run 输出一致性)")
    print("=" * 76)

    for framework in ["pytorch", "onnx"]:
        print(f"\n  {framework}:")
        for name in [p.name for p in IMAGE_PATHS]:
            run_outputs = []
            for run_idx, (pt_results, ort_results) in enumerate(all_runs):
                data = pt_results if framework == "pytorch" else ort_results
                run_outputs.append(data[name])

            ref = run_outputs[0]
            all_same = all(torch.equal(ref, o) for o in run_outputs[1:])

            if all_same:
                print(f"    {name:<20} 全部 {args.runs} 次输出完全一致")
            else:
                max_inter_diff = max(torch.abs(ref - o).max().item() for o in run_outputs[1:])
                print(f"    {name:<20} 存在差异  max_inter_diff={max_inter_diff:.2e}")

    # ---- 检查 3: 全局汇总 ----
    print("\n" + "=" * 76)
    print("  [3] 汇总")
    print("=" * 76)

    global_max_pt_ort = 0.0
    for pt_results, ort_results in all_runs:
        for name in [p.name for p in IMAGE_PATHS]:
            d = torch.abs(pt_results[name] - ort_results[name]).max().item()
            global_max_pt_ort = max(global_max_pt_ort, d)

    print(f"\n  PyTorch vs ONNX 全局最大误差: {global_max_pt_ort:.2e}")

    if global_max_pt_ort < 1e-4:
        print("  结论: ONNX 导出数值正确 (误差 < 1e-4)")
    elif global_max_pt_ort < 1e-2:
        print("  结论: ONNX 导出基本正确，存在微小数值差异 (误差 < 1e-2)")
    else:
        print("  结论: 差异较大，请检查导出配置")

    pt_repro = all(
        torch.equal(all_runs[0][0][n], run[0][n])
        for run in all_runs[1:]
        for n in [p.name for p in IMAGE_PATHS]
    )
    ort_repro = all(
        torch.equal(all_runs[0][1][n], run[1][n])
        for run in all_runs[1:]
        for n in [p.name for p in IMAGE_PATHS]
    )
    print(f"  PyTorch 可复现: {'是' if pt_repro else '否'}")
    print(f"  ONNX    可复现: {'是' if ort_repro else '否'}")


if __name__ == "__main__":
    main()
