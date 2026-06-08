.PHONY: onnx test all clean \
       venv export infer-all infer-all-torch infer-all-onnx verify \
       quantize quantize-verify quantize-int8 quantize-int8-verify \
       docker docker-up docker-shell docker-down \
       build-host cross-build prepare-rootfs prepare-app-files starry-link starry-test \
       sg2002-sdcard sg2002-clean

PYTHON   ?= .venv/bin/python
CARGO    ?= cargo

DOCKER_IMAGE ?= ghcr.io/rcore-os/tgoskits-container:latest
DOCKER_NAME  := starryos-act-infer

MODEL_PT    := output/train/model.pt
MODEL_ONNX  := output/train/model.onnx
MODEL_FP16  := output/train/model_fp16.onnx
MODEL_INT8  := output/train/model_int8.onnx

RISCV_TOOLCHAIN_ROOT ?= /opt/riscv64-linux-musl-cross

CROSS_TARGET := riscv64gc-unknown-linux-musl
CROSS_BIN    := act-infer-ort/target/$(CROSS_TARGET)/release/act-infer-ort

IMG_DIR := output/dataset/videos/observation.images.fpv/chunk-000

ROOTFS_BASE   := tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img
ROOTFS_APP    := tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-act-infer.img

RESULT_TORCH := output/infer_results_torch.json
RESULT_ONNX  := output/infer_results_onnx.json
RESULT_FP16  := output/infer_results_fp16.json
RESULT_INT8  := output/infer_results_int8.json

all: onnx test

# --- ONNX pipeline: venv → export → torch infer → onnx infer → verify ---

onnx: verify

export: venv $(MODEL_ONNX)

infer-all-torch: venv $(RESULT_TORCH)

infer-all-onnx: venv $(RESULT_ONNX)

infer-all: onnx

verify: $(RESULT_TORCH) $(RESULT_ONNX)
	$(PYTHON) scripts/verify_onnx.py

venv:
	uv pip install -r requirements.txt

$(MODEL_ONNX): scripts/export_onnx.py $(MODEL_PT)
	$(PYTHON) scripts/export_onnx.py

$(RESULT_TORCH): scripts/batch_infer_torch.py $(MODEL_PT)
	$(PYTHON) scripts/batch_infer_torch.py

$(RESULT_ONNX): scripts/batch_infer_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/batch_infer_onnx.py

# --- FP16 quantization pipeline ---

quantize: quantize-verify

$(MODEL_FP16): scripts/quantize_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/quantize_onnx.py

$(RESULT_FP16): scripts/batch_infer_onnx.py $(MODEL_FP16)
	$(PYTHON) scripts/batch_infer_onnx.py --model $(MODEL_FP16) --output $(RESULT_FP16)

quantize-verify: $(RESULT_ONNX) $(RESULT_FP16)
	$(PYTHON) scripts/verify_onnx.py --a $(RESULT_ONNX) --b $(RESULT_FP16)

# --- INT8 quantization pipeline ---

quantize-int8: quantize-int8-verify

$(MODEL_INT8): scripts/quantize_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/quantize_onnx.py --int8

$(RESULT_INT8): scripts/batch_infer_onnx.py $(MODEL_INT8)
	$(PYTHON) scripts/batch_infer_onnx.py --model $(MODEL_INT8) --output $(RESULT_INT8)

quantize-int8-verify: $(RESULT_ONNX) $(RESULT_INT8)
	$(PYTHON) scripts/verify_onnx.py --a $(RESULT_ONNX) --b $(RESULT_INT8)

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

host-test: build-host
	bash scripts/install-ort.sh
	$(HOST_BIN) --model $(MODEL_ONNX) \
		--dir $(IMG_DIR) \
		--stats output/dataset/meta/stats.json \
		--reference $(RESULT_ONNX)

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

# --- SG2002 SD card image ---

sg2002-sdcard:
	bash scripts/build-sg2002-sdcard.sh

sg2002-clean:
	sudo umount mnt/sg2002_rootfs 2>/dev/null || true
	rm -rf output/sg2002

# --- Clean ---

clean:
	rm -f $(MODEL_ONNX) $(MODEL_FP16) $(MODEL_INT8)
	rm -f output/infer_results_torch.json output/infer_results_onnx.json
	rm -f output/infer_results_fp16.json output/infer_results_int8.json
	rm -f $(STARRY_APP_LINK)
	rm -f starry-apps/act-infer/act-infer-ort
	rm -f starry-apps/act-infer/model.onnx
	rm -rf starry-apps/act-infer/frames
	rm -rf act-infer-ort/target
	rm -rf tgoskits/target
