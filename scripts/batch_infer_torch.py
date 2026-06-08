import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from act.configuration_act import ACTConfig
from act.modeling_act import ACTModel

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_checkpoint(path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        config_dict = ckpt.get("config", {})
        inf_mu = ckpt.get("inference_latent_mu")
        inf_log_sigma = ckpt.get("inference_latent_log_sigma")
    else:
        state_dict, config_dict = ckpt, {}
        inf_mu = inf_log_sigma = None

    config = ACTConfig(**config_dict)
    model = ACTModel(config)
    model.load_state_dict(state_dict)
    if inf_mu is not None:
        model.set_inference_latent(inf_mu, inf_log_sigma)
    model = model.to(device).eval()
    return model, config_dict


def load_stats(stats_dir):
    stats_path = Path(stats_dir) / "meta" / "stats.json"
    if not stats_path.exists():
        print("stats.json not found, using defaults")
        return {
            "state_q01": [0.0, 0.0],
            "state_q99": [1.0, 1.0],
            "action_q01": [0.0, 0.0, 0.0],
            "action_q99": [1.0, 1.0, 1.0],
        }
    with open(stats_path) as f:
        raw = json.load(f)
    return {
        "state_q01": raw["observation.state"]["q01"],
        "state_q99": raw["observation.state"]["q99"],
        "action_q01": raw["action"]["q01"],
        "action_q99": raw["action"]["q99"],
    }


def process_image(path, device):
    img = Image.open(path).convert("RGB")
    t = IMAGE_TRANSFORM(img)
    return t.unsqueeze(0).unsqueeze(0).to(device)


def normalize_state(state, stats, device):
    q01 = torch.tensor(stats["state_q01"], dtype=torch.float32, device=device)
    q99 = torch.tensor(stats["state_q99"], dtype=torch.float32, device=device)
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8, device=device), d)
    return 2 * (s - q01) / d - 1


def denormalize_action(action, stats, device):
    q01 = torch.tensor(stats["action_q01"], dtype=torch.float32, device=device)
    q99 = torch.tensor(stats["action_q99"], dtype=torch.float32, device=device)
    d = q99 - q01
    d = torch.where(d == 0, torch.tensor(1e-8, device=device), d)
    return (action + 1) / 2 * d + q01


def determine_turn(left_vel, right_vel):
    if left_vel < right_vel:
        return "LEFT"
    elif left_vel > right_vel:
        return "RIGHT"
    return "STRAIGHT"


def main():
    model_path = PROJECT_ROOT / "output" / "train" / "model.pt"
    stats_dir = PROJECT_ROOT / "output" / "dataset"
    img_dir = PROJECT_ROOT / "output" / "dataset" / "videos" / "observation.images.fpv" / "chunk-000"
    output_path = PROJECT_ROOT / "output" / "infer_results_torch.json"

    device = get_device()
    print(f"Device: {device}")

    model, config_dict = load_checkpoint(model_path, device)
    state_dim = config_dict.get("state_dim", 2)
    action_dim = config_dict.get("action_dim", 3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} | state_dim={state_dim} | action_dim={action_dim}")

    stats = load_stats(stats_dir)
    state = [0.0] * state_dim
    state_tensor = normalize_state(state, stats, device)

    images = sorted(img_dir.glob("*.jpg"))
    total = len(images)
    print(f"Found {total} images")

    dim_labels = ["left_vel", "right_vel", "gripper_target"][:action_dim]
    results = []
    t0 = time.time()

    with torch.no_grad():
        for i, img_path in enumerate(images):
            image_tensor = process_image(img_path, device)
            action_chunk = model.forward(image_tensor, state_tensor, action_target=None, infer_cvae=False)["action"]
            action_chunk_denorm = denormalize_action(action_chunk[0], stats, device)
            first_step = action_chunk_denorm[0].cpu().tolist()

            left_vel, right_vel = first_step[0], first_step[1]
            turn = determine_turn(left_vel, right_vel)

            results.append({
                "frame": img_path.name,
                "left_vel": round(first_step[0], 6),
                "right_vel": round(first_step[1], 6),
                "gripper_target": round(first_step[2], 6),
                "turn": turn,
            })

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  [{i+1}/{total}] {img_path.name} | {turn:>8s} | "
                      f"left={left_vel:+.6f} right={right_vel:+.6f} | "
                      f"elapsed={elapsed:.1f}s eta={eta:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {total} frames in {elapsed:.1f}s ({elapsed/total:.3f}s/frame)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")

    left_count = sum(1 for r in results if r["turn"] == "LEFT")
    right_count = sum(1 for r in results if r["turn"] == "RIGHT")
    straight_count = sum(1 for r in results if r["turn"] == "STRAIGHT")
    print(f"Summary: LEFT={left_count} RIGHT={right_count} STRAIGHT={straight_count}")


if __name__ == "__main__":
    main()
