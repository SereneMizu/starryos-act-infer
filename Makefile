.PHONY: all clean \
        export-onnx verify-onnx quantize quantize-int8 \
        docker-up docker-down docker-shell \
        tpu-up tpu-down tpu-shell \
        test-host test-qemu build-sg2002 sg2002-sdcard \
        build-rk3588 rk3588-sdcard \
        build-lrzsz

PYTHON := .venv/bin/python

# --- Containers ---

DOCKER_IMAGE ?= ghcr.io/rcore-os/tgoskits-container:latest
DOCKER_NAME  := starryos-act-infer
DOCKER_EXEC  := docker exec $(DOCKER_NAME)

TPU_IMAGE    ?= sophgo/tpuc_dev:latest
TPU_DOCKER_NAME     := tpu-mlir-act
TPU_DOCKER_EXEC := docker exec $(TPU_DOCKER_NAME)

# --- Cross-compilation ---

TARGET     := riscv64gc-unknown-linux-musl
TOOLCHAIN  ?= /opt/riscv64-linux-musl-cross
LINKER_ENV := CARGO_TARGET_$(shell echo $(TARGET) | tr 'a-z-' 'A-Z_')_LINKER=$(TOOLCHAIN)/bin/riscv64-linux-musl-gcc

# --- Paths ---

MODEL_PT     := output/train/model.pt
MODEL_ONNX   := output/train/model.onnx
MODEL_FP16   := output/train/model_fp16.onnx
MODEL_INT8   := output/train/model_int8.onnx
RESULT_TORCH := output/infer_results_torch.json
RESULT_ONNX  := output/infer_results_onnx.json
RESULT_FP16  := output/infer_results_fp16.json
RESULT_INT8  := output/infer_results_int8.json
IMG_DIR      := output/dataset/videos/observation.images.fpv/chunk-000
ROOTFS_BASE  := /tmp/.tgos-images/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img
ROOTFS_APP   := /tmp/.tgos-images/rootfs-riscv64-alpine.img/rootfs-riscv64-act-infer.img
ROOTFS_RK3588 := /tmp/.tgos-images/rootfs-aarch64-debian.img/rootfs-aarch64-debian.img
TGOSIMAGES   := https://github.com/rcore-os/tgosimages/releases/download/v0.0.7
SG2002_UIMG  := output/sg2002/starryos.uimg
SG2002_BOARD := os/StarryOS/configs/board/licheerv-nano-sg2002.toml
RK3588_UIMG  := output/rk3588/starryos.uimg
RK3588_BOARD := os/StarryOS/configs/board/orangepi-5-plus.toml
STARRY_LINK  := tgoskits/apps/starry/act-infer

all: export-onnx verify-onnx

# === Python pipeline (host) ===

export-onnx: $(MODEL_ONNX)

$(MODEL_ONNX): scripts/export_onnx.py $(MODEL_PT)
	$(PYTHON) scripts/export_onnx.py

$(RESULT_TORCH): scripts/batch_infer_torch.py $(MODEL_PT)
	$(PYTHON) scripts/batch_infer_torch.py

$(RESULT_ONNX): scripts/batch_infer_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/batch_infer_onnx.py

verify-onnx: $(RESULT_TORCH) $(RESULT_ONNX)
	$(PYTHON) scripts/verify_results.py --reference $(RESULT_TORCH) --result $(RESULT_ONNX)

$(MODEL_FP16): scripts/quantize_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/quantize_onnx.py

$(RESULT_FP16): scripts/batch_infer_onnx.py $(MODEL_FP16)
	$(PYTHON) scripts/batch_infer_onnx.py --model $(MODEL_FP16) --output $(RESULT_FP16)

quantize: $(RESULT_ONNX) $(RESULT_FP16)
	$(PYTHON) scripts/verify_results.py --reference $(RESULT_ONNX) --result $(RESULT_FP16)

$(MODEL_INT8): scripts/quantize_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/quantize_onnx.py --int8

$(RESULT_INT8): scripts/batch_infer_onnx.py $(MODEL_INT8)
	$(PYTHON) scripts/batch_infer_onnx.py --model $(MODEL_INT8) --output $(RESULT_INT8)

quantize-int8: $(RESULT_ONNX) $(RESULT_INT8)
	$(PYTHON) scripts/verify_results.py --reference $(RESULT_ONNX) --result $(RESULT_INT8)

# === Containers ===

docker-up:
	@if ! docker start $(DOCKER_NAME) 2>/dev/null; then \
		docker run -d --name $(DOCKER_NAME) --privileged -v "$$(pwd)":/workspace -w /workspace $(DOCKER_IMAGE) sleep infinity; \
		$(DOCKER_EXEC) apt-get update; \
		$(DOCKER_EXEC) apt-get install u-boot-tools fdisk parted -y; \
	fi

docker-shell: docker-up
	docker exec -it $(DOCKER_NAME) bash

docker-down:
	docker stop $(DOCKER_NAME) 2>/dev/null || true
	docker rm $(DOCKER_NAME) 2>/dev/null || true

tpu-up:
	@if ! docker start $(TPU_DOCKER_NAME) 2>/dev/null; then \
		docker run -d --privileged --name $(TPU_DOCKER_NAME) -v "$$(pwd)":/workspace -w /workspace $(TPU_IMAGE) sleep infinity; \
	fi
	$(TPU_DOCKER_EXEC) pip install tpu_mlir; \

tpu-shell: tpu-up
	docker exec -it $(TPU_DOCKER_NAME) bash

tpu-down:
	docker stop $(TPU_DOCKER_NAME) 2>/dev/null || true
	docker rm $(TPU_DOCKER_NAME) 2>/dev/null || true

# === Task 3: QEMU (CPU) ===

test-host: docker-up $(MODEL_ONNX)
	$(DOCKER_EXEC) bash -c 'cd /workspace/act-infer-ort && cargo build --release'
	$(DOCKER_EXEC) bash scripts/install-ort.sh
	$(DOCKER_EXEC) /workspace/act-infer-ort/target/release/act-infer-ort --model /workspace/$(MODEL_ONNX) --dir /workspace/$(IMG_DIR) --stats /workspace/output/dataset/meta/stats.json --reference /workspace/$(RESULT_ONNX)

test-qemu: docker-up $(MODEL_ONNX) $(ROOTFS_APP)
	ln -sfn ../../../starry-apps/act-infer $(STARRY_LINK)
	$(DOCKER_EXEC) bash -c 'cd /workspace/act-infer-ort && rustup default stable 2>/dev/null; rustup target add $(TARGET) 2>/dev/null; $(LINKER_ENV) cargo build --release --target $(TARGET)'
	bash scripts/prepare-app-files.sh
	$(DOCKER_EXEC) bash -c 'cd /workspace/tgoskits && cargo xtask starry app qemu -t act-infer --arch riscv64'

# === Task 2: RK3588 (OrangePi 5 Plus) ===

$(RK3588_UIMG): docker-up
	$(DOCKER_EXEC) bash -c 'cd /workspace/tgoskits && cargo xtask starry build --config $(RK3588_BOARD) --arch aarch64'
	@mkdir -p output/rk3588
	uimg_src=$$(find tgoskits/target/aarch64-unknown-linux-musl -name "*.uimg" 2>/dev/null | head -1); \
		[ -n "$$uimg_src" ] && cp "$$uimg_src" $@

build-rk3588: $(RK3588_UIMG)

rk3588-sdcard: build-rk3588 $(ROOTFS_RK3588)
	sudo bash scripts/build-rk3588-sdcard.sh

# === Task 1: SG2002 (TPU) ===

$(SG2002_UIMG): docker-up
	$(DOCKER_EXEC) bash -c 'cd /workspace/tgoskits && cargo xtask starry build --config $(SG2002_BOARD) --arch riscv64'
	@mkdir -p output/sg2002
	uimg_src=$$(find tgoskits/target/riscv64gc-unknown-linux-musl -name "*.uimg" 2>/dev/null | head -1); \
		[ -n "$$uimg_src" ] && cp "$$uimg_src" $@

build-sg2002: $(SG2002_UIMG) tpu-up $(MODEL_ONNX)
	$(TPU_DOCKER_EXEC) python scripts/tpu_compile.py --quantize BF16 --processor cv181x
	$(DOCKER_EXEC) bash -c 'rustup default stable 2>/dev/null; rustup target add $(TARGET) 2>/dev/null; cd /workspace/act-infer-tpu && $(LINKER_ENV) cargo build --release --target $(TARGET)'
	$(DOCKER_EXEC) riscv64-linux-musl-strip /workspace/act-infer-tpu/target/$(TARGET)/release/act-infer-tpu

SG2002_BOOT := sdboot/sg2002-boot.img
RK3588_BOOT := sdboot/rk3588-boot.img
# sdboot/sg2002-boot.img: 从 Sipeed LicheeRV-Nano 官方镜像 p1 分区提取
#   源: https://github.com/sipeed/LicheeRV-Nano-Build/releases/download/20260114/2026-01-14-16-03-d4003f.tar.xz
#   提取: dd if=<official.img> of=sdboot/sg2002-boot.img bs=512 skip=1 count=32768
# sdboot/rk3588-boot.img: 从 Armbian OrangePi 5 Plus 镜像前 16MB 提取（Rockchip SPL/ATF/U-Boot）
#   源: Armbian_26.5.1_Orangepi5-plus_trixie_current_6.18.33_minimal.img
#   提取: dd if=<armbian.img> of=sdboot/rk3588-boot.img bs=512 count=32768
# starry-apps/act-infer-tpu/lib/:
#   libcviruntime.so, libcvikernel.so, libcvimath.so: 从 Sipeed 官方镜像 rootfs /usr/bin/lib/ 提取
#   libgcc_s.so.1, libstdc++.so.6: 从 tgoskits 容器 /opt/riscv64-linux-musl-cross/riscv64-linux-musl/lib/ 提取

# === Misc tools (static, cross) ===

build-lrzsz: docker-up
	$(DOCKER_EXEC) bash scripts/build-lrzsz.sh

sg2002-sdcard: build-sg2002 verify-onnx $(ROOTFS_BASE)
	sudo bash scripts/build-sg2002-sdcard.sh

# === Rootfs ===

$(ROOTFS_BASE): docker-up
	$(DOCKER_EXEC) bash -c 'cd /workspace/tgoskits && cargo xtask starry rootfs --arch riscv64'
	@mkdir -p /tmp/.tgos-images/rootfs-riscv64-alpine.img
	docker cp $(DOCKER_NAME):/tmp/.tgos-images/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img $(ROOTFS_BASE)

$(ROOTFS_RK3588):
	@mkdir -p /tmp/.tgos-images/rootfs-aarch64-debian.img
	[ -f $@ ] || (curl -fSL $(TGOSIMAGES)/rootfs-aarch64-debian.img.tar.xz | tar xJ -C /tmp/.tgos-images/rootfs-aarch64-debian.img)

$(ROOTFS_APP): $(ROOTFS_BASE)
	cp $(ROOTFS_BASE) $(ROOTFS_APP)
	bash scripts/prepare-rootfs.sh $(ROOTFS_APP)

# === Clean ===

clean:
	docker stop $(DOCKER_NAME) $(TPU_DOCKER_NAME) 2>/dev/null || true
	docker rm $(DOCKER_NAME) $(TPU_DOCKER_NAME) 2>/dev/null || true
	rm -f $(STARRY_LINK)
	rm -rf starry-apps/act-infer/act-infer-ort starry-apps/act-infer/model.onnx starry-apps/act-infer/frames
	sudo rm -rf act-infer-ort/target act-infer-tpu/target
	sudo rm -rf output/sg2002 output/rk3588 output/tpu output/lrzsz output/infer_results_*.json mnt
	sudo rm -rf tgoskits/target tgoskits/tmp /tmp/.tgos-images
	sudo rm -rf third_party
