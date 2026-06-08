#!/usr/bin/env bash
# 将交叉编译产物（二进制、ONNX 模型、测试图片、归一化参数）拷贝到 starry-apps 打包目录
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dst="$root/starry-apps/act-infer"
ort_target="riscv64gc-unknown-linux-musl"

mkdir -p "$dst"

install -Dm0755 "$root/act-infer-ort/target/$ort_target/release/act-infer-ort" "$dst/act-infer-ort"
install -Dm0644 "$root/output/train/model.onnx" "$dst/model.onnx"
install -Dm0644 "$root/output/dataset/meta/stats.json" "$dst/stats.json"

rm -rf "$dst/frames"
mkdir -p "$dst/frames"
cp "$root/output/dataset/videos/observation.images.fpv/chunk-000"/*.jpg "$dst/frames/"
