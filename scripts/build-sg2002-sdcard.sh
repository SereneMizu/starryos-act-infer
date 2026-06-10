#!/usr/bin/env bash
# 构建 SG2002 (LicheeRV-Nano) SD 卡镜像
# 包含：boot 分区（官方 u-boot）+ rootfs 分区（tgoskits Alpine rootfs + StarryOS 内核 + TPU 推理应用）
# 用法: $0 <rootfs.img>  （tgoskits 管理的 ext4 rootfs 镜像）
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/sg2002"
mnt="$proj/mnt/sg2002_rootfs"

ROOTFS_IMG="${1:?Usage: $0 <rootfs.img>}"
OFFICIAL_IMG_URL="https://github.com/sipeed/LicheeRV-Nano-Build/releases/download/20260114/2026-01-14-16-03-d4003f.tar.xz"

TPU_BIN="$proj/act-infer-tpu/target/riscv64gc-unknown-linux-musl/release/act-infer-tpu"
TPU_CVMODEL="$proj/output/tpu/act_model_cv181x_bf16.cvimodel"
STATS_JSON="$proj/output/dataset/meta/stats.json"
FRAMES_DIR="$proj/output/dataset/videos/observation.images.fpv/chunk-000"
REF_JSON="$proj/output/infer_results_onnx.json"
INFER_SH="$proj/starry-apps/act-infer-tpu/infer.sh"
APP_DEST="/opt/act-infer"

[[ -f "$ROOTFS_IMG" ]] || { echo "[sg2002] rootfs not found: $ROOTFS_IMG"; exit 1; }
mkdir -p "$out"

official_tar="$out/official-img.tar.xz"
official_img="$out/official.img"
sdcard="$out/sg2002-sdcard.img"
loop_file="$out/.loop_dev"

# --- 下载官方镜像（含 u-boot）---

if [[ ! -f "$official_img" ]]; then
    [[ -f "$official_tar" ]] || { echo "[sg2002] downloading official image ..."; curl -fSL -o "$official_tar" "$OFFICIAL_IMG_URL"; }
    echo "[sg2002] extracting ..."
    tmp=$(mktemp -d)
    tar -xf "$official_tar" -C "$tmp"
    cp "$(find "$tmp" -name "*.img" | head -1)" "$official_img"
    rm -rf "$tmp"
fi

# --- 创建 2GB SD 卡镜像：p1=boot(16MB), p2=rootfs ---

rm -f "$sdcard"
fallocate -l 2147483648 "$sdcard"
printf "start=1, size=32768, type=c, bootable\nstart=32769, type=83\n" | sfdisk "$sdcard" > /dev/null

LOOP=$(losetup --find --show --partscan "$sdcard")
echo "$LOOP" > "$loop_file"

cleanup() {
    umount "$mnt" 2>/dev/null || true
    rmdir "$mnt" 2>/dev/null || true
    losetup -D "$LOOP" 2>/dev/null || true
    rm -f "$loop_file"
}
trap cleanup EXIT

if ! [ -b "${LOOP}p1" ]; then
    rm -f "${LOOP}p1" "${LOOP}p2" 2>/dev/null || true
    partprobe "$LOOP"
fi

BOOT="${LOOP}p1"
ROOT="${LOOP}p2"

# --- 写入 boot 分区 ---

dd if="$official_img" bs=512 skip=1 count=32768 of="$BOOT" status=none

# --- 写入 tgoskits rootfs（ext4 镜像直接 dd，然后扩展分区）---

echo "[sg2002] writing rootfs ..."
dd if="$ROOTFS_IMG" of="$ROOT" bs=4M status=none
e2fsck -f -y "$ROOT" >/dev/null 2>&1 || true
resize2fs "$ROOT" >/dev/null
mkdir -p "$mnt"
mount "$ROOT" "$mnt"

# --- StarryOS 内核（由 Makefile 预构建到 output/sg2002/starryos.uimg）---

echo "[sg2002] installing kernel ..."
cp "$out/starryos.uimg" "$mnt/starryos.uimg"
cp "$out/workspace_sg2002.uimg" "$mnt/workspace_sg2002.uimg" 2>/dev/null || true

# --- 安装 TPU 推理应用 ---

echo "[sg2002] installing app ..."
mkdir -p "$mnt/usr/bin" "$mnt$APP_DEST"

cp "$TPU_BIN" "$mnt/usr/bin/act-infer-tpu"
chmod +x "$mnt/usr/bin/act-infer-tpu"
cp "$TPU_CVMODEL" "$mnt$APP_DEST/model.cvimodel"
cp "$STATS_JSON" "$mnt$APP_DEST/stats.json"
cp -r "$FRAMES_DIR" "$mnt$APP_DEST/frames"
cp "$REF_JSON" "$mnt$APP_DEST/reference.json" 2>/dev/null || true
cp "$INFER_SH" "$mnt$APP_DEST/infer.sh"
chmod +x "$mnt$APP_DEST/infer.sh"

# --- 运行时动态库（从 starry-apps/act-infer-tpu/lib staging 收集）---

STAGING="$proj/starry-apps/act-infer-tpu/lib"

echo "[sg2002] installing runtime libs ..."
for so in "$STAGING"/*.so*; do
    [ -f "$so" ] || continue
    base=$(basename "$so")
    cp "$so" "$mnt/lib/$base"
    echo "  $base"
done
ln -sf libgcc_s.so.1 "$mnt/lib/libgcc_s.so" 2>/dev/null || true
ln -sf libstdc++.so.6 "$mnt/lib/libstdc++.so" 2>/dev/null || true

# --- 卸载并生成最终镜像 ---

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"


echo "[sg2002] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
