tgoskits是上游代码，应当避免修改（当前已在dev分支上有局部patch，见下方）

脚本内部禁用sudo，由Makefile在调用处加sudo提权（sg2002-sdcard、rk3588-sdcard、collect-sg2002-libs）

## 构建命令

- `make test-qemu` — QEMU仿真测试：交叉编译act-infer-ort→准备rootfs→xtask app qemu运行
- `make test-host` — 主机本机测试act-infer-ort（需要ort dylib）
- `make build-sg2002` — 编译内核 + TPU模型 + 交叉编译act-infer-tpu
- `make sg2002-sdcard` — build-sg2002 + 打包SD卡镜像
- `make build-rk3588` — 编译内核 + RKNN模型 + 交叉编译act-infer-rknn
- `make docker-up` / `make tpu-up` — 启动构建容器
- `make clean` — 停容器 + 清理产物

## Docker容器

- `starryos-act-infer`（ghcr.io/rcore-os/tgoskits-container，Ubuntu 24.04）— 内核编译、Rust交叉编译，内有 `/opt/riscv64-linux-musl-cross` 工具链
- `tpu-mlir-act`（sophgo/tpuc_dev）— TPU-MLIR模型编译（`model_transform.py` / `model_deploy.py`）
- 首次启动容器后需 `docker exec starryos-act-infer rustup toolchain install stable`（容器镜像可能不含预装工具链）
- docker-up 已配置 apt 清华镜像源，tpu-up 已配置 pip 清华镜像源
- 容器内 `/tmp` 重启即清空；不要依赖 `/tmp/.tgos-images` 路径

## QEMU 测试（act-infer-ort）

- 流程：docker-up → 下载alpine rootfs → chroot安装onnxruntime → 交叉编译ort → prepare-app-files.sh准备文件 → xtask app qemu运行
- rootfs在容器内编译产生，通过volume mount（`-v $(pwd):/workspace`）直接出现在 `tgoskits/tmp/axbuild/rootfs/`，不需要 `docker cp`
- `scripts/prepare-rootfs.sh`：chroot进alpine镜像，替换为清华alpine源后 `apk add onnxruntime`
- `scripts/prepare-app-files.sh`：拷贝二进制、ONNX模型、测试帧、stats.json到 `starry-apps/act-infer/`
- `starry-apps/act-infer/prebuild.sh`：xtask调用的注入脚本，将文件写入rootfs overlay（`$STARRY_OVERLAY_DIR`）
- 测试只跑2帧：`frame_000000.jpg`（LEFT）和 `frame_000227.jpg`（RIGHT）
- QEMU配置 `starry-apps/act-infer/qemu-riscv64.toml` 的drive路径必须用managed image名（如 `rootfs-riscv64-alpine.img`），否则xtask无法解析
- 测试结果与 `output/infer_results_torch.json` 对比验证
- 参考帧子集：`output/test/reference.json`（2条记录）

## 内存测量

- 三个act-infer-*程序统一使用 **MemFree差值法**：后台线程每10ms采样 `/proc/meminfo` 的MemFree，追踪最小值，peak = (baseline_free - min_free) / 1024
- **不要用VmSize**：StarryOS当前实现中VMRSS = VmSize（`mm/stats.rs:169` FIXME），虚拟地址空间不反映物理内存
- `/proc/meminfo` 的 MemTotal/MemFree/MemAvailable/Cached/AnonPages/PageTables 数据真实有效
- 模型加载时间、图片预处理时间、推理时间已在 `print_summary` 中分别输出

## tgoskits 子模块

- 当前跟踪 `feat/yhy` 分支（`.gitmodules` 的 `branch = feat/yhy`），remote 为 `SereneMizu/tgoskits`，上游为 `rcore-os/tgoskits`
- TPU/ION cfg gate 修复已提交到 `sg2002-plat-dyn-tpu-fix` 分支（`db56ec8f`），`feat/yhy` 基于它还多一个 RK3588 提交（`86f656ac1`，当前 HEAD）
- 修改内容：把 `#[cfg(all(feature = "sg2002", not(feature = "plat-dyn")))]` 改为 `#[cfg(feature = "sg2002")]`
  - 涉及文件：`file/mod.rs`、`pseudofs/dev/mod.rs`、`pseudofs/sysfs.rs`、`syscall/mm/mmap.rs`
  - 原因：上游 TPU/ION 设备注册只在 `plat_dyn=false`（静态平台）下生效，但静态平台在 SG2002 上启动崩溃（NULL函数指针）。改为 `plat_dyn=true`（动态平台，能正常启动）+ 放开 cfg gate 让 TPU 设备也注册
- **不要**尝试切到 `plat_dyn=false`，静态平台内核会在 SG2002 上 Instruction access fault
- 同步上游：`git -C tgoskits fetch upstream && git -C tgoskits merge upstream/dev`（冲突以远程为准：`git checkout --theirs <file>`）

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

- `act-infer-ort/src/main.rs` — ONNX Runtime CPU推理（QEMU测试用）
- `act-infer-tpu/src/main.rs` — TPU推理，BF16→FP32 输出转换，图像预处理
- `act-infer-tpu/src/bindings.rs` — bindgen生成的CVIRuntime FFI绑定（手动裁剪）
- `act-infer-tpu/build.rs` — 静态链接指令
- `scripts/tpu_compile.py` — ONNX→cvimodel 编译
- `scripts/build-sg2002-sdcard.sh` — SD卡镜像拼装
- `scripts/prepare-rootfs.sh` — chroot alpine安装onnxruntime
- `scripts/prepare-app-files.sh` — 准备QEMU测试文件
- `starry-apps/act-infer/{prebuild.sh,qemu-riscv64.toml,act-infer-test.sh}` — xtask app配置
- `sg2002-libs/` — 静态库 `.a` + `libgcc_s.so.1`

## 注意

- git提交信息用Conventional Commits风格：`feat:` / `refactor:` / `chore:` + 中文单行
- 不要手动运行 `cargo` 命令，统一用 `make` 目标
- Makefile中已配置apt（清华）、apk（清华）、pip（清华）国内镜像源
- make 命令用 Bash tool 执行（带足够 timeout），不要用 pty_spawn
