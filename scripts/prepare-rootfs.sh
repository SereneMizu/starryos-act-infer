#!/usr/bin/env bash
# 将 Alpine rootfs 镜像中安装 ONNX Runtime（chroot + apk）
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
rootfs_img="${1:?Usage: $0 <rootfs.img>}"
mnt="$proj/mnt/rootfs"

[[ -f "$rootfs_img" ]] || { echo "error: $rootfs_img not found" >&2; exit 1; }

mkdir -p "$mnt"
mount -o loop "$rootfs_img" "$mnt"

chroot "$mnt" /bin/sh -c '
  echo "https://mirrors.tuna.tsinghua.edu.cn/alpine/v3.23/main" > /etc/apk/repositories
  echo "https://mirrors.tuna.tsinghua.edu.cn/alpine/v3.23/community" >> /etc/apk/repositories
  apk add --no-cache onnxruntime
'
ln -sf libonnxruntime.so.1 "$mnt/usr/lib/libonnxruntime.so"

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true

echo "[prepare-rootfs] done: $rootfs_img"
