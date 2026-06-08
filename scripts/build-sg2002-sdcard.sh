#!/usr/bin/env bash
# 构建 SG2002 (LicheeRV-Nano) SD 卡镜像
# 包含：boot 分区（官方 u-boot）+ rootfs 分区（Alpine + StarryOS 内核 + TPU 推理应用）
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
out="$proj/output/sg2002"
mnt="$proj/mnt/sg2002_rootfs"

BOARD_CONFIG="os/StarryOS/configs/board/licheerv-nano-sg2002.toml"
OFFICIAL_IMG_URL="https://github.com/sipeed/LicheeRV-Nano-Build/releases/download/20260114/2026-01-14-16-03-d4003f.tar.xz"
ALPINE_ROOTFS_URL="https://mirrors.tuna.tsinghua.edu.cn/alpine/v3.23/releases/riscv64/alpine-minirootfs-3.23.4-riscv64.tar.gz"

TPU_BIN="$proj/act-infer-tpu/target/riscv64gc-unknown-linux-musl/release/act-infer-tpu"
TPU_CVMODEL="$proj/output/tpu/act_model_cv181x_bf16.cvimodel"
STATS_JSON="$proj/output/dataset/meta/stats.json"
FRAMES_DIR="$proj/output/dataset/videos/observation.images.fpv/chunk-000"
REF_JSON="$proj/output/infer_results_onnx.json"
INFER_SH="$proj/starry-apps/act-infer-tpu/infer.sh"
APP_DEST="/opt/act-infer"

mkdir -p "$out"

official_tar="$out/official-img.tar.xz"
official_img="$out/official.img"
alpine_tar="$out/alpine-minirootfs-3.23.4-riscv64.tar.gz"
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

# --- 下载 Alpine rootfs ---

[[ -f "$alpine_tar" ]] || { echo "[sg2002] downloading alpine rootfs ..."; curl -fSL -o "$alpine_tar" "$ALPINE_ROOTFS_URL"; }

# --- 创建 1GB SD 卡镜像：p1=boot(16MB), p2=rootfs ---

rm -f "$sdcard"
fallocate -l 1073741824 "$sdcard"
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

# --- 创建 rootfs 分区 ---

mkfs.ext4 -F -L rootfs "$ROOT" > /dev/null
mkdir -p "$mnt"
mount "$ROOT" "$mnt"
tar -xzf "$alpine_tar" -C "$mnt"

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

# --- C++ 运行时（libstdc++ 已静态链接，仅剩 libgcc_s 动态）---

echo "[sg2002] installing runtime libs ..."
cp "$proj/sg2002-libs/libgcc_s.so.1" "$mnt/lib/"
ln -sf libgcc_s.so.1 "$mnt/lib/libgcc_s.so"

# --- 卸载并生成最终镜像 ---

umount "$mnt"
rmdir "$mnt" 2>/dev/null || true
trap - EXIT
losetup -d "$LOOP" 2>/dev/null || true
rm -f "$loop_file"


echo "[sg2002] done: $sdcard ($(stat -c%s "$sdcard" | numfmt --to=iec))"
