# AGENTS.md — starryos-act-infer

## 仓库概述

ACT 模型嵌入式推理部署（赛题 Pro57）：SG2002 TPU / RK3588 NPU / QEMU CPU 三平台。

- 父分支 `board`，上游主分支 `main`。另有 `xiji` remote 用于提交。
- 子模块 `tgoskits`：跟踪 `feat/yhy` 分支（`SereneMizu/tgoskits`），上游 `rcore-os/tgoskits`。**tgoskits 应避免修改**（见下方 patch）。
- 文档：`docs/初赛文档.md`、`docs/sg2002.md`、`docs/rk3588.md`、`docs/quantization_v2.md`

## 铁律

- **脚本内部禁用 sudo**，由 Makefile 在调用处加 sudo（`sg2002-sdcard`、`rk3588-sdcard`）
- **不要手动运行 cargo**，统一用 `make` 目标
- **不要手动运行 docker exec python**，统一用 `make` 目标
- make 命令用 Bash tool 执行（带足够 timeout），**不要用 pty_spawn**

## 构建命令

```makefile
make docker-up           # → 启动构建容器（starryos-act-infer），自动 apt 装依赖
make tpu-up              # → 启动 TPU-MLIR 容器（tpu-mlir-act）
make test-host           # → 主机本机测试 act-infer-ort（需 ort dylib）
make test-qemu           # → QEMU 仿真：交叉编译 ort → rootfs → xtask app qemu
make model-onnx          # → PyTorch → FP32 ONNX
make model-onnx-mixed    # → ORT 混合精度量化 (enc_full 策略, 详见 docs/quantization_v2.md)
make model-sg2002-bf16   # → TPU BF16 cvimodel (全模型 BF16)
make model-sg2002-mixed  # → TPU INT8+BF16 cvimodel (enc_full 策略, decoder ffn/attn_out 保 BF16)
make build-sg2002        # → 内核 uImage + model-sg2002-mixed + 交叉编译 act-infer-tpu
make sg2002-sdcard       # → build-sg2002 + verify-onnx + rootfs → SD 卡镜像
make build-rk3588        # → 内核 uImage
make model-rk3588-fp16   # → RKNN FP16 模型 (ONNX FP16 → RKNN, 不量化)
make model-rk3588-mixed  # → RKNN INT8+FP16 混合精度模型 (已量化 ONNX 直转)
make build-rk3588-binary # → 交叉编译 act-infer-rknn (aarch64 glibc)
make rk3588-sdcard       # → build-rk3588 + model-rk3588-mixed + build-rk3588-binary + rootfs → SD 卡镜像
make clean              # → 停容器 + 清理所有产物
```

## 命名规则

- `model-<平台>-<精度>`：只编译模型产物（不编译内核/Rust 二进制）
- `build-<平台>`：完整构建（内核 + 模型 + Rust 二进制）
- `<平台>-sdcard`：SD 卡镜像（依赖 build 目标，自动全链）

## 关键顺序要求

1. `make test-qemu` 需要先 `make model-onnx`（即 `make all`）产生 `.onnx` 和参考结果
2. `make quant-all` 顺序：FP32 ONNX → 混合精度量化 → 推理 → 验证
3. `make sg2002-sdcard` / `make rk3588-sdcard` 内部自动处理全依赖链，直接运行即可

## Docker 容器

| 名称 | 镜像 | 用途 | 备注 |
|------|------|------|------|
| `starryos-act-infer` | `ghcr.io/rcore-os/tgoskits-container` (Ubuntu 24.04) | 内核编译、Rust 交叉编译 | 内含 `/opt/riscv64-linux-musl-cross`；首次需 `rustup toolchain install stable`；volume mount `-v $(pwd):/workspace` |
| `tpu-mlir-act` | `sophgo/tpuc_dev` | TPU-MLIR 模型编译 | pip 已配清华源 |

- docker-up 已配 apt 清华源，tpu-up 已配 pip 清华源
- 容器内 `/tmp` 重启即清空；不要依赖 `/tmp/.tgos-images`

## tgoskits 子模块

- 跟踪 `feat/yhy`（`.gitmodules` 的 `branch = feat/yhy`），remote `SereneMizu/tgoskits`，上游 `rcore-os/tgoskits`
- **TPU/ION cfg gate 局部 patch**（已提交到 `sg2002-plat-dyn-tpu-fix` 分支）：
  - 改动：`#[cfg(all(feature = "sg2002", not(feature = "plat-dyn")))]` → `#[cfg(feature = "sg2002")]`
  - 涉及：`file/mod.rs`、`pseudofs/dev/mod.rs`、`pseudofs/sysfs.rs`、`syscall/mm/mmap.rs`
  - 原因：`plat_dyn=false` 在 SG2002 上 Instruction access fault；`plat_dyn=true` 时 cfg gate 会阻止 TPU 设备注册
- **不要**切到 `plat_dyn=false`
- 同步上游：`git -C tgoskits fetch upstream && git -C tgoskits merge upstream/dev`（冲突 `--theirs`）
- 推送两个仓库：父仓库 `git push origin board` + tgoskits `git push origin feat/yhy`。SSH 需先 `ssh-keyscan github.com >> ~/.ssh/known_hosts`

## Board 配置

**SG2002**（`licheerv-nano-sg2002.toml`）：
- `plat_dyn = true`，features 含 `starry-kernel/sg2002`（TPU+ION+DMA+串口）、`axplat-dyn/thead-mae`、`ax-driver/{sg2002-placeholder,cvsd,serial}`
- **不要**加 `myplat`（与 `ax-hal/riscv64-sg2002` 冲突）
- **不要**加 `ax-feat/bus-mmio` 或 `ax-feat/driver-cvsd`（不存在）

## act-infer-tpu（SG2002 TPU 推理）

- 交叉编译目标 `riscv64gc-unknown-linux-musl`
- **动态链接** `starry-apps/act-infer-tpu/lib/{libcviruntime.so,libcvikernel.so,libcvimath.so}` + `libstdc++.so.6` + `libgcc_s.so.1`
- `build.rs` 中 bindgen 从 `include/cviruntime.h` **编译时生成** FFI 绑定到 `$OUT_DIR/bindings.rs`（非手动维护）
- 交叉编译时 clang `--sysroot=/opt/riscv64-linux-musl-cross/riscv64-linux-musl`

## TPU 模型编译

- **BF16**（`tpu_compile_bf16.py`）：全模型 BF16，model_transform → model_deploy
- **混合精度**（`tpu_compile_mixed.py`）：INT8 + decoder ffn/attn_out BF16
  - 流程：model_transform → run_calibration（生成 cali_table）→ `gen_tpu_qtable.py`（从 enc_full 策略生成 qtable）→ model_deploy --quantize INT8 --quantize_table
  - `gen_tpu_qtable.py` 从 MLIR 层名解析，复用 `quantize_mixed_onnx.py` 的 `enc_full` 分层策略（encoder 全 INT8，decoder ffn1/ffn2/attn_out BF16）
  - **不要用 search_qtable 自动搜索**：cos 相似度指标不等于动作预测准确率，全 INT8 cos=0.99 但准确率 <50%
- cvimodel 输入 fp32，输出 fp32

## act-infer-ort（QEMU CPU 推理）

- 流程：`docker-up` → xtask rootfs → `prepare-rootfs.sh`（chroot 装 onnxruntime）→ 交叉编译 → `prepare-app-files.sh` → xtask app qemu
- rootfs 通过 volume mount 出现在 `tgoskits/tmp/axbuild/rootfs/`，无需 `docker cp`
- `starry-apps/act-infer/prebuild.sh`：xtask 调用的注入脚本，通过 `$STARRY_OVERLAY_DIR` 写文件
- QEMU 测试 2 帧：`frame_000000.jpg`（LEFT）和 `frame_000227.jpg`（RIGHT）
- `qemu-riscv64.toml` 的 drive 路径必须用 managed image 名（如 `rootfs-riscv64-alpine.img`）
- 参考帧子集：`output/test/reference.json`（2 条记录）

## act-infer-rknn（RK3588 NPU 推理）

- 交叉编译目标 `aarch64-unknown-linux-gnu`（**glibc**，因 `librknnrt.so` 依赖 glibc）
- 链接器 `aarch64-linux-gnu-gcc`（容器内 apt 安装）
- RKNN 模型编译需独立 venv `.venv-rknn`（Python 3.12，rknn-toolkit2 仅 cp312 wheel）
- **混合精度路径**：已量化的 `model_mixed_enc_full.onnx`（含 QDQ 节点）直转 RKNN，`do_quantization=False`。RKNN 识别 QAT 模型，自动折叠 QDQ，保留 INT8+FP16 分布
- **不要用 `do_quantization=True`**：RKNN 的 w8a8 会强制 INT8 量化输入 images，action 输出差异仅 ~0.001 量级，被噪声淹没，准确率崩到 68%
- `optimization_level=2`（level 3 会触发影响精度的优化 pass）

## 内存测量（三平台统一）

- **MemFree 差值法**：后台线程每 10ms 采样 `/proc/meminfo` 的 MemFree，追踪最小值
- peak = (baseline_free - min_free) / 1024
- **不要用 VmSize**（StarryOS 中 VMRSS = VmSize，不反映物理内存）
- `/proc/meminfo` 的 MemTotal/MemFree/MemAvailable/Cached/AnonPages/PageTables 真实有效

## 关键路径关系

```
output/
├── train/model.pt                       # PyTorch checkpoint（预训练）
├── train/model.onnx                     # model-onnx 产物
├── train/model_mixed_enc_full.onnx      # model-onnx-mixed 产物 (ORT 混合精度)
├── infer_results_torch.json             # PyTorch 参考结果
├── infer_results_onnx.json              # ONNX 验证结果
├── test/reference.json                  # QEMU 测试参考子集（2 帧）
├── tpu/act_model_cv181x_bf16.cvimodel   # model-sg2002-bf16 产物
├── tpu/act_model_cv181x_mixed.cvimodel  # model-sg2002-mixed 产物 (INT8+BF16)
├── rknn/act_model_rk3588_fp16.rknn      # model-rk3588-fp16 产物
├── rknn/act_model_rk3588_mixed.rknn     # model-rk3588-mixed 产物 (INT8+FP16)
├── sg2002/sg2002-sdcard.img             # SG2002 SD 卡镜像（2G）
└── rk3588/rk3588-sdcard.img             # RK3588 SD 卡镜像（2G）
```

## SD 卡镜像结构

- SG2002: 分区1=boot（官方 u-boot）、分区2=ext4（Alpine musl）
  - `/usr/bin/act-infer-tpu`、`/opt/act-infer/{model.cvimodel,frames,stats.json,infer.sh}`、`/lib/libgcc_s.so.1`
- RK3588: 分区1=boot（Rockchip SPL/ATF/U-Boot）、分区2=ext4（Debian glibc）
  - `/usr/bin/act-infer-rknn`、`/opt/act-infer/{model.rknn,frames,stats.json,infer.sh}`、`/usr/lib/librknnrt.so`

## Python 流水线

- `PYTHON := .venv/bin/python`（主机 FP32/FP16 ONNX 流程）
- `RKNN_PYTHON := .venv-rknn/bin/python`（RKNN 编译，Python 3.12）
- CUDA EP 需要 `LD_LIBRARY_PATH` 包含 `site-packages/nvidia/*/lib/`
- 量化分析脚本：`scripts/{sensitivity_analysis,per_layer_sensitivity,calib_compare}.py`