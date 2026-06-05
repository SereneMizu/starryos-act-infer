.PHONY: export verify docker docker-up docker-shell docker-down \
       build-host cross-build prepare-rootfs prepare-app-files starry-link starry-test onnx test all clean

PYTHON   ?= .venv/bin/python
CARGO    ?= cargo

DOCKER_IMAGE ?= ghcr.io/rcore-os/tgoskits-container:latest
DOCKER_NAME  := starryos-act-infer

MODEL_PT   := output/train/model.pt
MODEL_ONNX := output/train/model.onnx

RISCV_TOOLCHAIN_ROOT ?= /opt/riscv64-linux-musl-cross

CROSS_TARGET := riscv64gc-unknown-linux-musl
CROSS_BIN    := act-infer-ort/target/$(CROSS_TARGET)/release/act-infer-ort

IMG_DIR := output/dataset/videos/observation.images.fpv/chunk-000
TEST_IMAGES := $(IMG_DIR)/frame_000000.jpg $(IMG_DIR)/frame_000227.jpg

ROOTFS_BASE   := tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img
ROOTFS_APP    := tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-act-infer.img

all: onnx test

onnx: export verify
test: host-test starry-test

# --- ONNX export & verify ---

export: $(MODEL_ONNX)

$(MODEL_ONNX): scripts/export_onnx.py $(MODEL_PT)
	$(PYTHON) scripts/export_onnx.py

verify: $(MODEL_ONNX)
	$(PYTHON) scripts/verify_onnx.py

# --- Docker ---


docker-up:
	docker run -d --name $(DOCKER_NAME) --privileged -v "$$(pwd)":/workspace -w /workspace $(DOCKER_IMAGE) sleep infinity

docker-shell:
	docker exec -it $(DOCKER_NAME) bash

docker-down:
	docker stop $(DOCKER_NAME) && docker rm $(DOCKER_NAME)

docker:
	docker run -it --rm -v "$$(pwd)":/workspace -w /workspace $(DOCKER_IMAGE)

# --- Host build & test (x86) ---

HOST_BIN := act-infer-ort/target/release/act-infer-ort

build-host:
	rustup default stable 2>/dev/null || true
	cd act-infer-ort && cargo build --release

host-test: build-host $(MODEL_ONNX)
	bash scripts/install-ort.sh
	$(HOST_BIN) --model $(MODEL_ONNX) \
		--left $(word 1,$(TEST_IMAGES)) \
		--right $(word 2,$(TEST_IMAGES))

# --- Cross-compile act-infer-ort ---

cross-build:
	rustup default stable 2>/dev/null || true
	cd act-infer-ort && \
		rustup target add $(CROSS_TARGET) && \
		CARGO_TARGET_$(shell echo $(CROSS_TARGET) | tr 'a-z-' 'A-Z_')_LINKER=$(RISCV_TOOLCHAIN_ROOT)/bin/riscv64-linux-musl-gcc \
		cargo build --release --target $(CROSS_TARGET)

# --- Rootfs prepare (chroot + apk install) ---

prepare-rootfs: $(ROOTFS_APP)

$(ROOTFS_APP): $(ROOTFS_BASE)
	cp $(ROOTFS_BASE) $(ROOTFS_APP)
	bash scripts/prepare-rootfs.sh $(ROOTFS_APP)

$(ROOTFS_BASE):
	cd tgoskits && $(CARGO) xtask starry rootfs --arch riscv64

# --- Prepare app resources ---

prepare-app-files: cross-build $(MODEL_ONNX)
	bash scripts/prepare-app-files.sh

# --- StarryOS test ---

STARRY_APP_LINK := tgoskits/apps/starry/act-infer

starry-link:
	ln -sfn "$$(pwd)/starry-apps/act-infer" $(STARRY_APP_LINK)

starry-test: starry-link prepare-app-files prepare-rootfs
	cd tgoskits && $(CARGO) xtask starry app qemu -t act-infer --arch riscv64

# --- Clean ---

clean:
	rm -f $(MODEL_ONNX)
	rm -f $(STARRY_APP_LINK)
	rm -f starry-apps/act-infer/act-infer-ort
	rm -f starry-apps/act-infer/model.onnx
	rm -f starry-apps/act-infer/frame_*.jpg
	rm -rf act-infer-ort/target
	rm -rf tgoskits/target
