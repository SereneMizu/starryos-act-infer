.PHONY: all clean \
        model-onnx verify-onnx \
        quant-sensitivity quant-per-layer quant-calib-compare \
        model-onnx-mixed quant-verify quant-all \
        docker-up docker-down docker-shell \
        tpu-up tpu-down tpu-shell \
        test-host test-qemu \
        model-sg2002-bf16 model-sg2002-mixed build-sg2002 sg2002-sdcard \
        build-rk3588 model-rk3588-fp16 model-rk3588-mixed rk3588-sdcard \
        build-lrzsz

PYTHON := .venv/bin/python
# RKNN 模型编译用独立 venv (3.12)：rknn-toolkit2 仅 cp312 wheel，且依赖 onnx 1.16.1（含 onnx.mapping）
RKNN_PYTHON := .venv-rknn/bin/python

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

# RKNN (RK3588) 交叉编译目标为 glibc：librknnrt.so 依赖 glibc (libc.so.6/libstdc++.so.6 等)，
# 不能用 musl，否则 musl+glibc 混链会冲突。容器内需 apt 装 gcc-aarch64-linux-gnu。
RKNN_TARGET     := aarch64-unknown-linux-gnu
RKNN_LINKER_ENV := CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

# --- Paths ---

MODEL_PT     := output/train/model.pt
MODEL_ONNX   := output/train/model.onnx
MODEL_FP16   := output/train/model_fp16.onnx
MODEL_MIXED  := output/train/model_mixed_enc_full.onnx
RESULT_TORCH := output/infer_results_torch.json
RESULT_ONNX  := output/infer_results_onnx.json
RESULT_MIXED := output/infer_results_mixed_enc_full.json
REF_TEST     := output/test/reference.json
IMG_DIR      := output/dataset/videos/observation.images.fpv/chunk-000
ROOTFS_BASE  := tgoskits/tmp/axbuild/rootfs/rootfs-riscv64-alpine.img/rootfs-riscv64-alpine.img
ROOTFS_ORT  := $(ROOTFS_BASE)
ROOTFS_RK3588 := /tmp/.tgos-images/rootfs-aarch64-debian.img/rootfs-aarch64-debian.img
TGOSIMAGES   := https://github.com/rcore-os/tgosimages/releases/download/v0.0.7
SG2002_UIMG  := output/sg2002/starryos.uimg
SG2002_BOARD := os/StarryOS/configs/board/licheerv-nano-sg2002.toml
RK3588_UIMG  := output/rk3588/starryos.uimg
RK3588_BOARD := os/StarryOS/configs/board/orangepi-5-plus.toml
RKNN_FP16    := output/rknn/act_model_rk3588_fp16.rknn
RKNN_MIXED   := output/rknn/act_model_rk3588_mixed.rknn
TPU_CVIMODEL_BF16   := output/tpu/act_model_cv181x_bf16.cvimodel
TPU_CVIMODEL_MIXED  := output/tpu/act_model_cv181x_mixed.cvimodel
STARRY_LINK  := tgoskits/apps/starry/act-infer

# CUDA EP 环境已移除: 改用纯 CPU onnxruntime (AVX-VNNI 加速 INT8).

all: model-onnx verify-onnx

# === Python pipeline (host) ===

model-onnx: $(MODEL_ONNX)

$(MODEL_ONNX): scripts/export_onnx.py $(MODEL_PT)
	$(PYTHON) scripts/export_onnx.py

$(RESULT_TORCH): scripts/batch_infer_torch.py $(MODEL_PT)
	$(PYTHON) scripts/batch_infer_torch.py

$(RESULT_ONNX): scripts/batch_infer_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/batch_infer_onnx.py

$(REF_TEST): scripts/extract_test_reference.py $(RESULT_TORCH)
	$(PYTHON) scripts/extract_test_reference.py

verify-onnx: $(RESULT_TORCH) $(RESULT_ONNX)
	$(PYTHON) scripts/verify_results.py --reference $(RESULT_TORCH) --result $(RESULT_ONNX)

# === Mixed-precision quantization (INT8 + FP16) ===
# 详见 docs/quantization_v2.md. CPU EP (AVX-VNNI 加速 INT8 GEMM).
# 策略 (enc_full): encoder 全 INT8 (conv+qkv+ffn+attn_out) + decoder 仅 qkv INT8,
#   decoder ffn/attn 敏感保 FP16. 比 conv+qkv 快 19%, 精度 98.8%.

# 按类别敏感度分析 (5 类, 666 帧, CPU)
quant-sensitivity:
	$(PYTHON) scripts/sensitivity_analysis.py

# 逐层 leave-one-in 敏感度分析 (conv + attn_out, 找类内坏分子)
quant-per-layer:
	$(PYTHON) scripts/per_layer_sensitivity.py

# 校准数据采样策略对比 (uniform/all/threshold 等 6 种)
quant-calib-compare:
	$(PYTHON) scripts/calib_compare.py

# 生成最终混合精度模型: enc_full (encoder 全 INT8 + decoder 仅 qkv INT8; conv 仅 encoder 有)
$(MODEL_MIXED): scripts/quantize_mixed_onnx.py $(MODEL_ONNX)
	$(PYTHON) scripts/quantize_mixed_onnx.py --preset enc_full --calib-mode all

model-onnx-mixed: $(MODEL_MIXED)

# batch_infer 666 帧 -> infer_results (5 轮测速取中位数)
$(RESULT_MIXED): scripts/batch_infer_onnx.py $(MODEL_MIXED)
	$(PYTHON) scripts/batch_infer_onnx.py --model $(MODEL_MIXED) --output $(RESULT_MIXED) --rounds 5

# verify_results 对比 FP32 参考
quant-verify: $(RESULT_MIXED) $(RESULT_ONNX)
	$(PYTHON) scripts/verify_results.py --reference $(RESULT_ONNX) --result $(RESULT_MIXED)

# 一键: 生成模型 -> 推理 -> 验证
quant-all: model-onnx-mixed quant-verify

# === Containers ===

docker-up:
	@if ! docker start $(DOCKER_NAME) 2>/dev/null; then \
		docker run -d --name $(DOCKER_NAME) --privileged -v "$$(pwd)":/workspace -w /workspace $(DOCKER_IMAGE) sleep infinity; \
		$(DOCKER_EXEC) bash -c 'sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null; true'; \
		$(DOCKER_EXEC) apt-get update; \
		$(DOCKER_EXEC) apt-get install u-boot-tools fdisk parted libclang-dev gcc-aarch64-linux-gnu -y; \
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
	$(TPU_DOCKER_EXEC) pip install tpu_mlir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; \

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

test-qemu: docker-up $(MODEL_ONNX) $(REF_TEST) $(ROOTFS_ORT)
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

# ONNX FP32 → FP16 预处理（主机 .venv）
$(MODEL_FP16): $(MODEL_ONNX)
	$(PYTHON) scripts/convert_onnx_fp16.py

# RKNN FP16 模型（主机 .venv-rknn，rknn-toolkit2）
$(RKNN_FP16): $(MODEL_FP16)
	$(RKNN_PYTHON) scripts/rknn_compile.py --target rk3588

model-rk3588-fp16: $(RKNN_FP16)

# RKNN 混合精度 (enc_full: QDQ INT8 + FP16): 已量化 ONNX 直转, 不重新量化
$(RKNN_MIXED): scripts/rknn_compile.py $(MODEL_MIXED)
	$(RKNN_PYTHON) scripts/rknn_compile.py --onnx $(MODEL_MIXED) --output $(RKNN_MIXED) --target rk3588

model-rk3588-mixed: $(RKNN_MIXED)

# RKNN Rust 交叉编译（容器，aarch64 glibc）
build-rk3588-binary: docker-up
	$(DOCKER_EXEC) bash -c 'rustup default stable 2>/dev/null; rustup target add $(RKNN_TARGET) 2>/dev/null; cd /workspace/act-infer-rknn && $(RKNN_LINKER_ENV) cargo build --release --target $(RKNN_TARGET)'
	$(DOCKER_EXEC) aarch64-linux-gnu-strip /workspace/act-infer-rknn/target/$(RKNN_TARGET)/release/act-infer-rknn

rk3588-sdcard: build-rk3588 model-rk3588-mixed build-rk3588-binary $(ROOTFS_RK3588)
	sudo bash scripts/build-rk3588-sdcard.sh

# === Task 1: SG2002 (TPU) ===

$(SG2002_UIMG): docker-up
	$(DOCKER_EXEC) bash -c 'cd /workspace/tgoskits && cargo xtask starry build --config $(SG2002_BOARD) --arch riscv64'
	@mkdir -p output/sg2002
	uimg_src=$$(find tgoskits/target/riscv64gc-unknown-linux-musl -name "*.uimg" 2>/dev/null | head -1); \
		[ -n "$$uimg_src" ] && cp "$$uimg_src" $@

# TPU BF16 cvimodel (全模型 BF16, 无 INT8)
$(TPU_CVIMODEL_BF16): tpu-up $(MODEL_ONNX)
	$(TPU_DOCKER_EXEC) python scripts/tpu_compile_bf16.py --quantize BF16 --processor cv181x

model-sg2002-bf16: $(TPU_CVIMODEL_BF16)

# TPU 混合精度 cvimodel: INT8 + BF16 (enc_full 策略: encoder 全 INT8, decoder ffn+attn_out BF16)
$(TPU_CVIMODEL_MIXED): tpu-up $(MODEL_ONNX)
	$(TPU_DOCKER_EXEC) python scripts/tpu_compile_mixed.py --processor cv181x --cali-num 100 --skip-verify

model-sg2002-mixed: $(TPU_CVIMODEL_MIXED)

# SG2002 Rust 交叉编译（容器，riscv64 musl）
build-sg2002-binary: $(SG2002_UIMG) docker-up
	$(DOCKER_EXEC) bash -c 'rustup default stable 2>/dev/null; rustup target add $(TARGET) 2>/dev/null; cd /workspace/act-infer-tpu && $(LINKER_ENV) cargo build --release --target $(TARGET)'
	$(DOCKER_EXEC) riscv64-linux-musl-strip /workspace/act-infer-tpu/target/$(TARGET)/release/act-infer-tpu

build-sg2002: $(SG2002_UIMG) model-sg2002-mixed build-sg2002-binary

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
	sudo bash scripts/prepare-rootfs.sh $@

$(ROOTFS_RK3588):
	@mkdir -p /tmp/.tgos-images/rootfs-aarch64-debian.img
	[ -f $@ ] || (curl -fSL $(TGOSIMAGES)/rootfs-aarch64-debian.img.tar.xz | tar xJ -C /tmp/.tgos-images/rootfs-aarch64-debian.img)

# === Clean ===

clean:
	docker stop $(DOCKER_NAME) $(TPU_DOCKER_NAME) 2>/dev/null || true
	docker rm $(DOCKER_NAME) $(TPU_DOCKER_NAME) 2>/dev/null || true
	rm -f $(STARRY_LINK)
	rm -rf starry-apps/act-infer/act-infer-ort starry-apps/act-infer/model.onnx starry-apps/act-infer/frames
	sudo rm -rf act-infer-ort/target act-infer-tpu/target act-infer-rknn/target
	sudo rm -rf output/sg2002 output/rk3588 output/rknn output/tpu output/lrzsz output/infer_results_*.json mnt
	sudo rm -rf tgoskits/target tgoskits/tmp/axbuild/starry-app /tmp/.tgos-images
	sudo rm -rf third_party