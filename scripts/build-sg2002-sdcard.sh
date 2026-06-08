#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
tgoskits_dir="$project_root/tgoskits"
out_dir="$project_root/output/sg2002"
mnt_rootfs="$project_root/mnt/sg2002_rootfs"

OFFICIAL_IMG_URL="https://github.com/sipeed/LicheeRV-Nano-Build/releases/download/20260114/2026-01-14-16-03-d4003f.tar.xz"
ALPINE_ROOTFS_URL="https://mirrors.tuna.tsinghua.edu.cn/alpine/v3.23/releases/riscv64/alpine-minirootfs-3.23.4-riscv64.tar.gz"
BOARD_CONFIG="os/StarryOS/configs/board/licheerv-nano-sg2002.toml"

mkdir -p "$out_dir"

official_tar="$out_dir/official-img.tar.xz"
official_img="$out_dir/official.img"
alpine_tar="$out_dir/alpine-minirootfs-3.23.4-riscv64.tar.gz"
uimg="$out_dir/starryos.uimg"
sdcard="$out_dir/sg2002-sdcard.img"

if [[ ! -d "$tgoskits_dir/.git" ]]; then
    echo "error: tgoskits submodule not initialized. Run: git submodule update --init tgoskits" >&2
    exit 1
fi

# --- download official image ---
if [[ ! -f "$official_img" ]]; then
    if [[ ! -f "$official_tar" ]]; then
        echo "[sg2002] downloading official image ..."
        curl -fSL -o "$official_tar" "$OFFICIAL_IMG_URL"
    fi
    echo "[sg2002] extracting official.img ..."
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT
    tar -xf "$official_tar" -C "$tmp_dir"
    found_img="$(find "$tmp_dir" -name "official.img" | head -1)"
    if [[ -z "$found_img" ]]; then
        echo "error: official.img not found in archive" >&2
        exit 1
    fi
    cp "$found_img" "$official_img"
    rm -rf "$tmp_dir"
    trap - EXIT
fi

# --- download alpine rootfs ---
if [[ ! -f "$alpine_tar" ]]; then
    echo "[sg2002] downloading alpine rootfs ..."
    curl -fSL -o "$alpine_tar" "$ALPINE_ROOTFS_URL"
fi

# --- build uimg ---
if [[ ! -f "$uimg" ]]; then
    echo "[sg2002] building uimg ..."
    cd "$tgoskits_dir"
    cargo xtask starry build --config "$BOARD_CONFIG" --arch riscv64
    found_uimg="$(find "$tgoskits_dir" -name "starryos_riscv64*.uimg" -newer "$tgoskits_dir/Cargo.lock" 2>/dev/null | head -1)"
    if [[ -z "$found_uimg" ]]; then
        found_uimg="$(find "$tgoskits_dir/target" -name "*.uimg" -newer "$tgoskits_dir/Cargo.lock" 2>/dev/null | head -1)"
    fi
    if [[ -z "$found_uimg" ]]; then
        echo "error: could not find generated .uimg file" >&2
        exit 1
    fi
    cp "$found_uimg" "$uimg"
fi

# --- build sdcard image ---
echo "[sg2002] creating 2GB sdcard image ..."
sudo rm -f "$sdcard"
truncate -s 2147483648 "$sdcard"

echo "[sg2002] partitioning ..."
printf "start=1, size=32768, type=c, bootable\nstart=32769, type=83\n" \
    | sudo sfdisk "$sdcard" > /dev/null

loop_dev_file="$out_dir/.loop_dev"
sudo losetup --find --show --partscan "$sdcard" > "$loop_dev_file"
LOOP=$(cat "$loop_dev_file")

cleanup() {
    sudo umount "$mnt_rootfs" 2>/dev/null || true
    sudo rmdir "$mnt_rootfs" 2>/dev/null || true
    sudo losetup -d "$LOOP" 2>/dev/null || true
    rm -f "$loop_dev_file"
}
trap cleanup EXIT

echo "[sg2002] writing boot partition ..."
sudo dd if="$official_img" bs=512 skip=1 count=32768 of="${LOOP}p1" status=none

echo "[sg2002] creating ext4 rootfs ..."
sudo mkfs.ext4 -F -L rootfs "${LOOP}p2" > /dev/null

echo "[sg2002] populating rootfs ..."
sudo mkdir -p "$mnt_rootfs"
sudo mount "${LOOP}p2" "$mnt_rootfs"
sudo tar -xzf "$alpine_tar" -C "$mnt_rootfs"
sudo cp "$uimg" "$mnt_rootfs/starryos.uimg"

echo "[sg2002] finalizing ..."
sudo umount "$mnt_rootfs"
sudo rmdir "$mnt_rootfs"
sudo losetup -d "$LOOP"
rm -f "$loop_dev_file"
trap - EXIT

echo "[sg2002] done: $sdcard"
