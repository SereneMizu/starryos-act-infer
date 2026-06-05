#!/usr/bin/env bash
set -euo pipefail

rootfs_img="${1:?Usage: $0 <rootfs.img>}"
mountpoint="/tmp/act-infer-rootfs"

if [[ ! -f "$rootfs_img" ]]; then
    echo "error: $rootfs_img not found" >&2
    exit 1
fi

mkdir -p "$mountpoint"
trap 'umount "$mountpoint" 2>/dev/null; rmdir "$mountpoint" 2>/dev/null' EXIT

mount -o loop "$rootfs_img" "$mountpoint"

chroot "$mountpoint" /bin/sh -c 'apk add --no-cache onnxruntime'
ln -sf libonnxruntime.so.1 "$mountpoint/usr/lib/libonnxruntime.so"

umount "$mountpoint"
rmdir "$mountpoint" 2>/dev/null || true
trap - EXIT

echo "[prepare-rootfs] done: $rootfs_img"
