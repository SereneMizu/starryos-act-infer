tgoskits是上游代码，应当避免修改（当前已在dev分支上有局部patch，见下方）

makefile和脚本禁用sudo，假设权限充足，如需提权，使用sudo make

## 构建命令

- `make sg2002-sdcard` — 完整流程：编译内核 + TPU模型编译 + 交叉编译Rust推理程序 + 打包SD卡镜像
- `make build-sg2002` — 只编译，不打包镜像
- `make docker-up` / `make tpu-up` — 启动构建容器
- `make clean` — 停容器 + 清理产物

## Docker容器

- `starryos-act-infer`（ghcr.io/rcore-os/tgoskits-container）— 内核编译、Rust交叉编译，内有 `/opt/riscv64-linux-musl-cross` 工具链
- `tpu-mlir-act`（sophgo/tpuc_dev）— TPU-MLIR模型编译（`model_transform.py` / `model_deploy.py`）

## tgoskits 子模块

- 当前跟踪 `origin/dev` 分支（不再是 sg2002-tpu-v0.1.3 tag）
- **有4个文件的本地patch**（通过 `git diff` 在 tgoskits 目录可见）：
  - 把 `#[cfg(all(feature = "sg2002", not(feature = "plat-dyn")))]` 改为 `#[cfg(feature = "sg2002")]`
  - 涉及文件：`file/mod.rs`、`pseudofs/dev/mod.rs`、`pseudofs/sysfs.rs`、`syscall/mm/mmap.rs`
  - 原因：上游 TPU/ION 设备注册只在 `plat_dyn=false`（静态平台）下生效，但静态平台在 SG2002 上启动崩溃（NULL函数指针）。改为 `plat_dyn=true`（动态平台，能正常启动）+ 放开 cfg gate 让 TPU 设备也注册
- **不要**尝试切到 `plat_dyn=false`，静态平台内核会在 SG2002 上 Instruction access fault

## Board配置

- `tgoskits/os/StarryOS/configs/board/licheerv-nano-sg2002.toml`
- `plat_dyn = true`，features 含 `starry-kernel/sg2002`（拉入 TPU 驱动 + ION + DMA + 串口）+ `axplat-dyn/thead-mae` + `ax-driver/{sg2002-placeholder,cvsd,serial}`
- **不要**加 `myplat`：与 `ax-hal/riscv64-sg2002`（由 sg2002 feature 引入）冲突
- **不要**加 `ax-feat/bus-mmio` 或 `ax-feat/driver-cvsd`：当前版本不存在这些 feature

## TPU推理程序链接（act-infer-tpu）

- 静态链接 `sg2002-libs/libcviruntime-static.a`、`libcvikernel-static.a`、`libcvimath-static.a`
- 静态链接 `libstdc++`，动态链接 `libgcc_s.so.1`（musl工具链无 `libgcc.a`）
- 交叉编译目标：`riscv64gc-unknown-linux-musl`
- `build.rs` 中配置链接参数，不要改用动态 `.so`
- 使用 bindgen 生成的 `bindings.rs` 直接调用 `CVI_NN_*`，不使用 `libloading`

## TPU模型编译

- `scripts/tpu_compile.py` 调用 `model_transform.py` + `model_deploy.py`
- 参数：`--quantize BF16 --processor cv181x`（无 `--addr_mode`、`--num_core`）
- 产物：`output/tpu/act_model_cv181x_bf16.cvimodel`（~96MB）
- cvimodel 输入 fp32，内部 BF16 计算，输出 fp32（`action_Add_f32`）
- 无 test data 时自动加 `--skip_validation`

## SD卡镜像

- `output/sg2002/sg2002-sdcard.img`（1GB）
- 分区1：boot（官方镜像dd写入），分区2：ext4 rootfs（Alpine musl）
- rootfs 内：`/usr/bin/act-infer-tpu`、`/opt/act-infer/{model.cvimodel,frames,stats.json,infer.sh}`、`/lib/libgcc_s.so.1`

## 关键文件

- `act-infer-tpu/src/main.rs` — Rust TPU推理，BF16→FP32 输出转换，图像预处理
- `act-infer-tpu/src/bindings.rs` — bindgen 生成的 CVIRuntime FFI 绑定（手动裁剪，不含完整 cviruntime.h）
- `act-infer-tpu/build.rs` — 静态链接指令
- `scripts/tpu_compile.py` — ONNX→cvimodel 编译
- `scripts/build-sg2002-sdcard.sh` — SD 卡镜像拼装
- `sg2002-libs/` — 静态库 `.a` + `libgcc_s.so.1`

## 注意

- git提交信息保持简洁，单行中文commit信息