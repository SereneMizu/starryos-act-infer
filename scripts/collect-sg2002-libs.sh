#!/usr/bin/env bash
# 从各来源收集 SG2002 运行时动态库到 staging 目录
# 仅收集 .so，不收集 .a
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
staging="$proj/starry-apps/act-infer-tpu/lib"
official_img="$proj/output/sg2002/official.img"
mnt="/tmp/official_lib_mnt"
cvi_lib_path="usr/bin/lib"
docker_name="starryos-act-infer"
toolchain_lib="/opt/riscv64-linux-musl-cross/riscv64-linux-musl/lib"

mkdir -p "$staging"

# --- 从 official.img 提取 CVI 运行时 .so ---

if [[ ! -f "$official_img" ]]; then
    echo "[collect-libs] official.img not found, run 'make sg2002-sdcard' first to download it"
    exit 1
fi

echo "[collect-libs] extracting CVI libs from official.img ..."
LOOP=$(losetup --find --show --partscan "$official_img")
partprobe "$LOOP" 2>/dev/null || true
sleep 1
mkdir -p "$mnt"
mount -o ro "${LOOP}p2" "$mnt"

for lib in libcviruntime.so libcvikernel.so libcvimath.so; do
    src="$mnt/$cvi_lib_path/$lib"
    if [[ -f "$src" ]]; then
        cp "$src" "$staging/$lib"
        echo "  $lib"
    else
        echo "  WARNING: $lib not found in official.img"
    fi
done

umount "$mnt" 2>/dev/null || true
rmdir "$mnt" 2>/dev/null || true
losetup -d "$LOOP" 2>/dev/null || true

# --- 从 tgoskits docker 容器提取 libgcc_s.so.1 ---

echo "[collect-libs] extracting libgcc_s.so.1 from $docker_name ..."
if ! docker inspect "$docker_name" >/dev/null 2>&1; then
    echo "[collect-libs] container $docker_name not running, run 'make docker-up' first"
    exit 1
fi

docker exec "$docker_name" cp "$toolchain_lib/libgcc_s.so.1" "/workspace/starry-apps/act-infer-tpu/lib/libgcc_s.so.1"
echo "  libgcc_s.so.1"
docker exec "$docker_name" cp "$toolchain_lib/libstdc++.so.6" "/workspace/starry-apps/act-infer-tpu/lib/libstdc++.so.6"
echo "  libstdc++.so.6"

echo "[collect-libs] done: $(ls "$staging")"
