#!/usr/bin/env bash
# 在容器内安装 ONNX Runtime x64（动态库 + ldconfig）
set -euo pipefail

version="${1:-1.26.0}"
install_dir="/usr/lib"

ldconfig -p 2>/dev/null | grep -q libonnxruntime.so && { echo "[install-ort] already installed, skipping."; exit 0; }

echo "[install-ort] installing ONNX Runtime ${version} ..."
cd /tmp
curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v${version}/onnxruntime-linux-x64-${version}.tgz" -o onnxruntime.tgz
tar xzf onnxruntime.tgz
cp "onnxruntime-linux-x64-${version}/lib/libonnxruntime.so"* "${install_dir}/"
ldconfig
rm -rf "onnxruntime-linux-x64-${version}" onnxruntime.tgz
echo "[install-ort] done."
