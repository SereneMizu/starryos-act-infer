#!/usr/bin/env bash
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/sg2002"
mnt="$proj/mnt/sg2002_rootfs"

ROOTFS_IMG="$proj/tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img"
BOOT_IMG="$proj/sdboot/sg2002-boot.img"

TPU_BIN="$proj/act-infer-tpu/target/riscv64gc-unknown-linux-musl/release/act-infer-tpu"
TPU_CVMODEL="$proj/output/tpu/act_model_cv181x_bf16.cvimodel"
STATS_JSON="$proj/output/dataset/meta/stats.json"
FRAMES_DIR="$proj/output/dataset/videos/observation.images.fpv/chunk-000"
REF_JSON="$proj/output/infer_results_onnx.json"
INFER_SH="$proj/starry-apps/act-infer-tpu/infer.sh"
APP_DEST="/opt/act-infer"

[[ -f "$ROOTFS_IMG" ]] || { echo "[sg2002] rootfs not found: $ROOTFS_IMG"; exit 1; }
[[ -f "$BOOT_IMG" ]] || { echo "[sg2002] boot not found: $BOOT_IMG"; exit 1; }
mkdir -p "$out"

sdcard="$out/sg2002-sdcard.img"
loop_file="$out/.loop_dev"

rm -f "$sdcard"
fallocate -l 2G "$sdcard"
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

dd if="$BOOT_IMG" of="$BOOT" bs=512 status=none

echo "[sg2002] writing rootfs ..."
dd if="$ROOTFS_IMG" of="$ROOT" bs=4M status=none
e2fsck -f -y "$ROOT" >/dev/null 2>&1 || true
resize2fs "$ROOT" >/dev/null
mkdir -p "$mnt"
mount "$ROOT" "$mnt"

echo "[sg2002] installing kernel ..."
cp "$out/starryos.uimg" "$mnt/starryos.uimg"
cp "$out/workspace_sg2002.uimg" "$mnt/workspace_sg2002.uimg" 2>/dev/null || true

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

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"

echo "[sg2002] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
