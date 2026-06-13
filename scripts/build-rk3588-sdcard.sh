#!/usr/bin/env bash
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/rk3588"
mnt="$proj/mnt/rk3588_rootfs"

ROOTFS_IMG="/tmp/.tgos-images/rootfs-aarch64-debian.img/rootfs-aarch64-debian.img"
BOOTCHAIN="$proj/sdboot/rk3588-boot.img"
# 用 tgoskits 自带 DTB（compatible=rockchip,rk3588-rknpu，匹配内核 RKNPU 驱动）；
# Armbian 的 sdboot/rk3588-orangepi-5-plus.dtb 用的是 rknn-core 新 binding，probe 不触发。
DTB_SRC="$proj/tgoskits/os/StarryOS/configs/board/orangepi-5-plus.dtb"

RKNN_BIN="$proj/act-infer-rknn/target/aarch64-unknown-linux-gnu/release/act-infer-rknn"
RKNN_MODEL="$proj/output/rknn/act_model_rk3588.rknn"
RKNN_LIB="$proj/starry-apps/act-infer-rknn/lib/librknnrt.so"
STATS_JSON="$proj/output/dataset/meta/stats.json"
FRAMES_DIR="$proj/output/dataset/videos/observation.images.fpv/chunk-000"
REF_JSON="$proj/output/infer_results_onnx.json"
INFER_SH="$proj/starry-apps/act-infer-rknn/infer.sh"
APP_DEST="/opt/act-infer"

[[ -f "$ROOTFS_IMG" ]] || { echo "[rk3588] rootfs not found: $ROOTFS_IMG"; exit 1; }
[[ -f "$BOOTCHAIN" ]] || { echo "[rk3588] boot not found: $BOOTCHAIN"; exit 1; }
[[ -f "$DTB_SRC" ]] || { echo "[rk3588] dtb not found: $DTB_SRC"; exit 1; }
[[ -f "$RKNN_BIN" ]] || { echo "[rk3588] app not found: $RKNN_BIN (run 'make build-rknn')"; exit 1; }
[[ -f "$RKNN_MODEL" ]] || { echo "[rk3588] model not found: $RKNN_MODEL"; exit 1; }
[[ -f "$RKNN_LIB" ]] || { echo "[rk3588] librknnrt.so not found: $RKNN_LIB"; exit 1; }
mkdir -p "$out"

sdcard="$out/rk3588-sdcard.img"
loop_file="$out/.loop_dev"

rm -f "$sdcard"
fallocate -l 2G "$sdcard"

printf "label: gpt\nstart=32768, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4\n" | sfdisk "$sdcard" > /dev/null

dd if="$BOOTCHAIN" of="$sdcard" bs=512 skip=34 seek=34 count=$((32768 - 34)) conv=notrunc status=none

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
    rm -f "${LOOP}p1" 2>/dev/null || true
    partprobe "$LOOP"
    sleep 1
fi

ROOT="${LOOP}p1"

echo "[rk3588] writing rootfs ..."
dd if="$ROOTFS_IMG" of="$ROOT" bs=4M status=none
e2fsck -f -y "$ROOT" >/dev/null 2>&1 || true
resize2fs "$ROOT" >/dev/null
mkdir -p "$mnt"
mount "$ROOT" "$mnt"

echo "[rk3588] installing kernel ..."
cp "$out/starryos.uimg" "$mnt/starryos.uimg"

echo "[rk3588] installing dtb ..."
cp "$DTB_SRC" "$mnt/"

echo "[rk3588] installing app ..."
mkdir -p "$mnt/usr/bin" "$mnt$APP_DEST"
cp "$RKNN_BIN" "$mnt/usr/bin/act-infer-rknn"
chmod +x "$mnt/usr/bin/act-infer-rknn"
cp "$RKNN_MODEL" "$mnt$APP_DEST/model.rknn"
cp "$STATS_JSON" "$mnt$APP_DEST/stats.json"
cp -r "$FRAMES_DIR" "$mnt$APP_DEST/frames"
cp "$REF_JSON" "$mnt$APP_DEST/reference.json" 2>/dev/null || true
cp "$INFER_SH" "$mnt$APP_DEST/infer.sh"
chmod +x "$mnt$APP_DEST/infer.sh"

echo "[rk3588] installing RKNN runtime lib ..."
cp "$RKNN_LIB" "$mnt/lib/librknnrt.so"
echo "  librknnrt.so"

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"

echo "[rk3588] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
