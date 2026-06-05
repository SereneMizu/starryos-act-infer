#!/usr/bin/env bash
set -euo pipefail

app_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
overlay_dir="${STARRY_OVERLAY_DIR:-}"

if [[ -z "$overlay_dir" ]]; then
    echo "error: STARRY_OVERLAY_DIR is required" >&2
    exit 1
fi

dst="$overlay_dir/opt/act-infer"
mkdir -p "$dst"

install -Dm0755 "$app_dir/act-infer-ort" "$dst/act-infer-ort"
install -Dm0755 "$app_dir/act-infer-test.sh" "$dst/act-infer-test.sh"
install -Dm0644 "$app_dir/model.onnx" "$dst/model.onnx"
install -Dm0644 "$app_dir/frame_000000.jpg" "$dst/frame_000000.jpg"
install -Dm0644 "$app_dir/frame_000227.jpg" "$dst/frame_000227.jpg"
