#!/usr/bin/env bash
# 构建 RK3588 (OrangePi 5 Plus) SD 卡镜像
# 布局：0~16MB Rockchip 引导链(rk3588-bootchain.img), 16MB起 GPT rootfs(tgoskits Debian rootfs + StarryOS)
# 用法: $0 <rootfs.img>  （tgoskits 管理的 ext4 rootfs 镜像）
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/rk3588"
mnt="$proj/mnt/rk3588_rootfs"

ROOTFS_IMG="${1:?Usage: $0 <rootfs.img>}"
BOOTCHAIN="$out/rk3588-bootchain.img"

[[ -f "$ROOTFS_IMG" ]] || { echo "[rk3588] rootfs not found: $ROOTFS_IMG"; exit 1; }
[[ -f "$BOOTCHAIN" ]] || { echo "[rk3588] bootchain not found: $BOOTCHAIN"; echo "  run 'dd if=<armbian.img> of=$BOOTCHAIN bs=512 count=32768' first"; exit 1; }
mkdir -p "$out"

sdcard="$out/rk3588-sdcard.img"
loop_file="$out/.loop_dev"

# --- 创建 1GB SD 卡镜像 ---

rm -f "$sdcard"
fallocate -l 1073741824 "$sdcard"

# GPT 分区表：p1 从扇区 32768 开始（与 Armbian 布局一致）
printf "label: gpt\nstart=32768, type=0FC63DAF-8483-4772-8E79-3D69D8477DE4\n" | sfdisk "$sdcard" > /dev/null

# 写入 Rockchip 引导链，跳过 GPT 头（扇区 0~33）避免覆盖分区表
# 扇区 0: Protective MBR, 扇区 1~33: GPT header + partition entries
# 扇区 34~32767: SPL/ATF/U-Boot 等引导数据
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

# --- 写入 tgoskits rootfs（ext4 镜像直接 dd，然后扩展分区）---

echo "[rk3588] writing rootfs ..."
dd if="$ROOTFS_IMG" of="$ROOT" bs=4M status=none
e2fsck -f -y "$ROOT" >/dev/null 2>&1 || true
resize2fs "$ROOT" >/dev/null
mkdir -p "$mnt"
mount "$ROOT" "$mnt"

# --- StarryOS 内核 ---

echo "[rk3588] installing kernel ..."
cp "$out/starryos.uimg" "$mnt/starryos.uimg"

# --- 卸载并生成最终镜像 ---

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"

echo "[rk3588] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
