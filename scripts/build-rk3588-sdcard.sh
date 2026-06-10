#!/usr/bin/env bash
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/rk3588"
mnt="$proj/mnt/rk3588_rootfs"

ROOTFS_IMG="/tmp/.tgos-images/rootfs-aarch64-debian.img/rootfs-aarch64-debian.img"
BOOTCHAIN="$proj/sdboot/rk3588-boot.img"
DTB="$proj/sdboot/rk3588-orangepi-5-plus.dtb"

[[ -f "$ROOTFS_IMG" ]] || { echo "[rk3588] rootfs not found: $ROOTFS_IMG"; exit 1; }
[[ -f "$BOOTCHAIN" ]] || { echo "[rk3588] boot not found: $BOOTCHAIN"; exit 1; }
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

if [[ -f "$DTB" ]]; then
    echo "[rk3588] installing dtb ..."
    cp "$DTB" "$mnt/$(basename "$DTB")"
fi

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"

echo "[rk3588] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
