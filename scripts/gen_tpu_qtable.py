"""Generate TPU-MLIR qtable from enc_full strategy (mirrors quantize_mixed_onnx.py PRESETS["enc_full"]).

策略 (与 quantize_mixed_onnx.py 完全一致):
  conv:          all INT8 (encoder ResNet18)
  ffn1:          encoder INT8, decoder BF16
  ffn2:          encoder INT8, decoder BF16
  qkv:           all INT8 (self_attn/multihead_attn QKV 投影)
  attn_out_proj: encoder INT8, decoder BF16
  attn_core (MatMul_3/MatMul_4): 不量化 (softmax 附近, 保持默认)
  tiny (action_head/state_encoder/latent_proj): 不量化

用法 (在 tpu-mlir 容器内):
  python scripts/gen_tpu_qtable.py
  python scripts/gen_tpu_qtable.py --mlir output/tpu/act_model.mlir --output output/tpu/act_qtable_enc_full
"""
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MLIR = PROJECT_ROOT / "output" / "tpu" / "act_model.mlir"
DEFAULT_QTABLE = PROJECT_ROOT / "output" / "tpu" / "act_qtable_enc_full"


def generate_qtable(mlir_path: Path, qtable_path: Path):
    content = mlir_path.read_text()

    loc_to_name = {}
    for m in re.finditer(r'#loc(\d+)\s*=\s*loc\("([^"]*)"\)', content):
        loc_to_name[m.group(1)] = m.group(2)

    ops = []
    for m in re.finditer(r'%(\d+)\s*=\s*"top\.(\w+)".*?loc\(#loc(\d+)\)', content):
        op_type = m.group(2)
        name = loc_to_name.get(m.group(3))
        if name:
            ops.append((op_type, name))

    bf16_layers = []
    for op_type, name in ops:
        if op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        if "action_head" in name or "state_encoder" in name or "latent_proj" in name:
            continue
        if "MatMul_3" in name or "MatMul_4" in name:
            continue
        if "/decoder/" not in name:
            continue
        is_ffn = ("linear1" in name) or ("linear2" in name)
        is_attn_out = ("self_attn/Gemm" in name) or ("multihead_attn/Gemm" in name)
        if is_ffn or is_attn_out:
            bf16_layers.append(name)

    with open(qtable_path, "w") as f:
        f.write("# generated time: manual enc_full strategy\n")
        f.write("# strategy: encoder all INT8, decoder ffn+attn_out BF16, decoder qkv INT8\n")
        f.write("# source: quantize_mixed_onnx.py PRESETS[enc_full]\n")
        f.write("###\n")
        f.write("# op_name   quantize_mode\n")
        for name in bf16_layers:
            f.write(f"{name} BF16\n")

    print(f"[qtable] {len(bf16_layers)} BF16 layers -> {qtable_path}")
    for n in bf16_layers:
        print(f"  BF16: {n}")
    return bf16_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir", default=str(DEFAULT_MLIR))
    parser.add_argument("--output", default=str(DEFAULT_QTABLE))
    args = parser.parse_args()
    generate_qtable(Path(args.mlir), Path(args.output))