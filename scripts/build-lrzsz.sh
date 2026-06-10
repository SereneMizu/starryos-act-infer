#!/usr/bin/env bash
# 在 tgoskits 容器内用 musl 交叉工具链静态构建 lrzsz（orig 上游源码）
# 产物：output/lrzsz/<arch>/{lrz,lsz} + rz/sz/rb/rx/sb/sx 符号链接
#        output/lrzsz/lrzsz-static-<version>.tar.gz
set -euo pipefail

proj="$(cd "$(dirname "$0")/.." && pwd)"
third_party="$proj/third_party/lrzsz"
out_dir="$proj/output/lrzsz"

VERSION="0.12.21rc"
ORIG_TARBALL="lrzsz_${VERSION}.orig.tar.gz"
SRC_SUBDIR="lrzsz-${VERSION}"
ORIG_URL="https://mirrors.tuna.tsinghua.edu.cn/debian/pool/main/l/lrzsz/${ORIG_TARBALL}"

# 架构列表：triple -> qemu-user 二进制名
ARCHES=(
    "riscv64-linux-musl:qemu-riscv64-static"
    "aarch64-linux-musl:qemu-aarch64-static"
)

mkdir -p "$third_party" "$out_dir"

# --- 确保有下载工具 ---
download() { # url dest
    if command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    elif command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$2" "$1"
    else
        echo "[lrzsz] installing wget ..." >&2
        apt-get update -qq && apt-get install -y -qq wget ca-certificates >/dev/null
        wget -q -O "$2" "$1"
    fi
}

# --- 下载并解压 orig 源码到 third_party/lrzsz ---
if [[ ! -d "$third_party/$SRC_SUBDIR" ]]; then
    if [[ ! -f "$third_party/$ORIG_TARBALL" ]]; then
        echo "[lrzsz] downloading $ORIG_TARBALL ..."
        download "$ORIG_URL" "$third_party/$ORIG_TARBALL"
    fi
    echo "[lrzsz] extracting ..."
    tar xzf "$third_party/$ORIG_TARBALL" -C "$third_party"
fi

# --- 单架构构建 ---
build_arch() { # triple qemu
    local triple="$1" qemu="$2"
    local arch="${triple%%-*}"            # riscv64 / aarch64
    local tc="/opt/${triple}-cross"
    local tc_cc="$tc/bin/${triple}-gcc"
    local tc_readelf="$tc/bin/${triple}-readelf"
    local src="$third_party/$SRC_SUBDIR"
    local bld="$third_party/build-${arch}"

    [[ -x "$tc_cc" ]] || { echo "[lrzsz] missing toolchain: $tc_cc (run in tgoskits container)" >&2; return 1; }

    echo "[lrzsz] building $arch (static) ..."
    rm -rf "$bld" && cp -a "$src" "$bld"
    (
        cd "$bld"
        CC="$tc_cc" CFLAGS="-static -O2 -std=gnu17" LDFLAGS="-static" \
            ./configure --host="$triple" --disable-shared --enable-static \
            --disable-nls --disable-rpath --without-included-regex --prefix=/usr \
            >/dev/null
        make -j"$(nproc)" >/dev/null
    )

    # --- 验证：架构 + 静态链接 + qemu 运行 ---
    local bin="$bld/src/lrz"
    echo -n "  verify $arch: "
    "$tc_readelf" -h "$bin" | awk '/Machine/{m=$1} /Machine/{print $2, $3}' | head -1
    "$tc_readelf" -l "$bin" | grep -q "INTERP" && { echo "  ERROR: $arch has INTERP (not static)" >&2; return 1; }
    "$tc_readelf" -d "$bin" | grep -q "NEEDED" && { echo "  ERROR: $arch has NEEDED (not static)" >&2; return 1; }
    if command -v "$qemu" >/dev/null 2>&1; then
        echo -n "  run: "; "$qemu" "$bin" --version 2>&1 | head -1
    fi

    # --- 收集产物 + 符号链接 ---
    local dest="$out_dir/$arch"
    rm -rf "$dest" && mkdir -p "$dest"
    cp "$bld/src/lrz" "$bld/src/lsz" "$dest/"
    ( cd "$dest" && ln -sf lrz rz && ln -sf lrz rb && ln -sf lrz rx \
                 && ln -sf lsz sz && ln -sf lsz sb && ln -sf lsz sx )
    echo "  -> $dest"
}

for entry in "${ARCHES[@]}"; do
    build_arch "${entry%%:*}" "${entry##*:}"
done

# --- 打包 ---
echo "[lrzsz] packaging ..."
tarball="$out_dir/lrzsz-static-${VERSION}.tar.gz"
( cd "$out_dir" && tar czf "$(basename "$tarball")" riscv64 aarch64 )
echo "[lrzsz] done: $tarball"
