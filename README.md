# StarryOS ACT 推理

在 StarryOS（riscv64 QEMU）上部署 ACT（Action Chunking with Transformers）模型的推理流水线。

## 做了什么

将 ACT 模型从 PyTorch 导出为 ONNX 格式，编写 Rust 推理程序，交叉编译为 riscv64 musl 二进制，集成到 StarryOS 并在 QEMU 中成功运行推理。完整流水线：**图像预处理 → 模型推理 → 动作后处理**，左转/右转方向判断与参考实现一致。

## 验证结果

```
[ort] model loaded in 16296ms
[frame_000000.jpg] left_vel=-0.001582  right_vel=+0.007263  gripper=+0.000000
  turn: LEFT (correct)
[frame_000227.jpg] left_vel=+0.006104  right_vel=-0.000444  gripper=+0.000000
  turn: RIGHT (correct)

ACT_INFER_OK
```

| 指标 | 值 |
|------|-----|
| 可执行文件 `act-infer-ort`（riscv64 musl） | 3.2 MB |
| ONNX 模型 `model.onnx` | 194 MB |
| QEMU 内存配置 | 1 GB |
| 模型加载耗时（QEMU riscv64） | 16,296 ms |
| 完整推理流水线耗时 | 120 秒超时内完成 |
| frame_000000（左转） | `right_vel(+0.007263) > left_vel(-0.001582)` → LEFT |
| frame_000227（右转） | `left_vel(+0.006104) > right_vel(-0.000444)` → RIGHT |

方向判断逻辑：左轮速度 < 右轮速度 → 车体左转，反之右转。源码见 `act-infer-ort/src/main.rs:96-106`。

## 快速复现

### 前置条件

- 主机：Docker、Python 3.14
- 训练产物：`output/train/model.pt` 和 `output/dataset/` 已存在

### 步骤

```bash
# 1. 导出 ONNX 并验证（主机）
uv venv .venv && source .venv/bin/activate
uv pip install torch torchvision onnx onnxruntime onnxscript pillow numpy
make export          # model.pt → model.onnx
make verify          # PyTorch vs ONNX 数值一致性验证

# 2. 启动构建容器（主机）
make docker-up

# 3. x86 推理测试（容器内）
docker exec starryos-act-infer make -C /workspace host-test
# → 输出 ACT_INFER_OK

# 4. StarryOS riscv64 推理测试（容器内）
docker exec starryos-act-infer make -C /workspace starry-test
# → 自动：交叉编译 → 准备 rootfs → QEMU 启动 → 推理 → ACT_INFER_OK

# 5. 清理
make docker-down && make clean
```

`starry-test` 一步完成：交叉编译 riscv64 musl → 打包 app 文件 → 构建 rootfs（apk 安装 onnxruntime）→ 启动 QEMU → 运行推理。

### Make 目标速查

| 目标 | 说明 |
|------|------|
| `make export` | PyTorch → ONNX 导出 |
| `make verify` | 数值一致性验证 |
| `make docker-up` | 启动构建容器 |
| `make host-test` | x86 构建 + 推理（容器内） |
| `make starry-test` | riscv64 交叉编译 + QEMU 测试（容器内） |
| `make docker-down` | 停止容器 |
| `make clean` | 清理产物 |

## 项目结构

```
├── act/                    ACT 模型定义（组织方提供）
├── act-infer-ort/          Rust ONNX Runtime 推理程序
│   └── src/main.rs         推理主程序（预处理/推理/后处理/方向判定）
├── scripts/
│   ├── export_onnx.py      PyTorch → ONNX 导出（ACTInferenceWrapper）
│   ├── verify_onnx.py      PyTorch vs ONNX 数值对比
│   ├── install-ort.sh      容器内安装 ONNX Runtime
│   ├── prepare-rootfs.sh   rootfs 注入 ONNX Runtime（chroot + apk）
│   └── prepare-app-files.sh 复制交叉编译产物
├── starry-apps/act-infer/  StarryOS 应用 case
│   ├── prebuild.sh         rootfs overlay 注入
│   ├── act-infer-test.sh   运行时测试脚本
│   ├── qemu-riscv64.toml   QEMU 配置 + 测试判定正则
│   └── build-riscv64gc-unknown-none-elf.toml  内核构建配置
├── Makefile                构建流水线
├── output/                 训练产物（model.pt、model.onnx、数据集）
└── tgoskits/               StarryOS 基础设施（git submodule）
```

## 技术方案

**ONNX 导出**：通过 `ACTInferenceWrapper` 将 CVAE 随机采样替换为零 latent，转为确定性推理模型，`torch.onnx.export` + `dynamo=True` 导出 opset 18。

**交叉编译**：目标 `riscv64gc-unknown-linux-musl`，使用 `ort` crate 的 `load-dynamic` 特性运行时动态加载 `libonnxruntime.so`，通过 Alpine apk 安装 riscv64 版本。

**StarryOS 集成**：通过软链接 `starry-apps/act-infer/` → `tgoskits/apps/starry/act-infer` 将 app 接入 tgoskits 框架，通过 `cargo xtask starry app qemu -t act-infer --arch riscv64` 一键构建内核、准备 rootfs、启动 QEMU 并运行推理。

## AI 使用说明

我对 ACT 模型尚不熟悉，因此模型相关的工作暂时借助 AI 完成。构建系统、rootfs 准备、StarryOS 集成等工程性工作由我完成。使用的工具为 OpenCode，主要模型为 GLM-5.1。

### AI 辅助的部分

- **Rust 推理程序** `act-infer-ort/src/main.rs`：ONNX Runtime 调用、图像预处理（resize + ImageNet 归一化）、动作反归一化、左转/右转判定
- **ONNX 导出** `scripts/export_onnx.py`：`ACTInferenceWrapper` 设计、零 latent 确定性推理策略、CVAE 随机采样消除
- **ONNX 验证** `scripts/verify_onnx.py`：PyTorch vs ONNX 数值对比逻辑

### 我完成的部分

- 项目方案设计（ONNX Runtime + StarryOS riscv64）
- Makefile 构建流水线
- `scripts/prepare-rootfs.sh`（chroot + apk 注入 ONNX Runtime）
- `scripts/prepare-app-files.sh`（交叉编译产物打包）
- `starry-apps/act-infer/` 下全部配置（prebuild.sh、qemu-riscv64.toml、内核构建配置、测试脚本）
- Docker 构建容器方案与交叉编译工具链配置
- QEMU 环境调试与推理验证
- AI 生成代码的审查与修改

### 借鉴说明

- `starry-apps/act-infer/` 下的 `prebuild.sh`、`qemu-riscv64.toml`、`build-riscv64gc-unknown-none-elf.toml` 等配置文件借鉴了 `tgoskits/apps/starry/` 中已有 app 的写法
- `tgoskits/` 为 StarryOS 基础设施（git submodule），属于项目依赖

## 依赖

| 组件 | 位置 |
|------|------|
| Python 3.14（torch、onnx、onnxruntime） | 主机 |
| Docker | 主机 |
| Rust（riscv64gc-unknown-linux-musl） | 容器内置 |
| riscv64 musl 交叉工具链 | 容器内置 |
| QEMU riscv64 | 容器内置 |
| ONNX Runtime 1.26.0 | 容器内按需下载 |
