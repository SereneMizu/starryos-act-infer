#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
app_dir="$project_root/starry-apps/act-infer"
dst="$app_dir"

mkdir -p "$dst"

install -Dm0755 "$project_root/act-infer-ort/target/riscv64gc-unknown-linux-musl/release/act-infer-ort" "$dst/act-infer-ort"
install -Dm0644 "$project_root/output/train/model.onnx" "$dst/model.onnx"

img_dir="$project_root/output/dataset/videos/observation.images.fpv/chunk-000"
install -Dm0644 "$img_dir/frame_000000.jpg" "$dst/frame_000000.jpg"
install -Dm0644 "$img_dir/frame_000227.jpg" "$dst/frame_000227.jpg"
