#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${1:-1.26.0}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
INSTALL_DIR="/usr/lib"

if ldconfig -p 2>/dev/null | grep -q libonnxruntime.so; then
    echo "[install-ort] ONNX Runtime already installed, skipping."
    exit 0
fi

echo "[install-ort] Installing ONNX Runtime ${ORT_VERSION}..."
cd /tmp
curl -sL "${ORT_URL}" -o onnxruntime.tgz
tar xzf onnxruntime.tgz
cp "onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so"* "${INSTALL_DIR}/"
ldconfig
rm -rf "onnxruntime-linux-x64-${ORT_VERSION}" onnxruntime.tgz
echo "[install-ort] Done."
