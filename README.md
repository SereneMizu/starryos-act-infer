# StarryOS ACT 推理

在 StarryOS 上运行 ACT（Action Chunking with Transformers）模型的 ONNX 推理。

## 项目结构

```
├── act/                  ACT 模型定义与训练代码
├── act-infer-ort/        Rust ONNX Runtime 推理程序（ort）
├── output/               训练产物（model.pt、model.onnx、数据集）
├── scripts/
│   ├── export_onnx.py    PyTorch → ONNX 导出
│   ├── verify_onnx.py    PyTorch vs ONNX 数值验证
│   ├── install-ort.sh    容器内安装 ONNX Runtime 共享库
│   ├── prepare-rootfs.sh 向 rootfs 注入 ONNX Runtime（apk）
│   └── prepare-app-files.sh  复制交叉编译产物到 app 目录
├── starry-apps/act-infer/ StarryOS 应用 case（qemu/prebuild/测试脚本）
└── tgoskits/             StarryOS 基础设施（git submodule）
```

## 全流程

### 前置条件

```bash
# Python 虚拟环境（主机）
uv venv .venv
source .venv/bin/activate
uv pip install torch torchvision onnx onnxruntime onnxscript pillow numpy

# 数据集和模型文件（output/train/model.pt 必须存在）
```

### 1. ONNX 导出与验证 —— 主机执行

```bash
make export          # model.pt → model.onnx
make verify          # PyTorch vs ONNX 数值一致性（5 次验证）
```

### 2. 启动 Docker 构建容器

```bash
make docker-up       # 启动守护容器（ghcr.io/rcore-os/tgoskits-container）
```

### 3. x86 推理测试 —— 容器内执行

首次运行会自动下载 ONNX Runtime v1.26.0 共享库。

```bash
docker exec starryos-act-infer make -C /workspace host-test
```

成功输出 `ACT_INFER_OK`。

### 4. StarryOS riscv64 推理测试 —— 容器内执行

```bash
docker exec starryos-act-infer make -C /workspace starry-test
```

该目标自动完成：交叉编译 riscv64 musl → 准备 app 文件 → 构建 rootfs（apk 安装 onnxruntime）→ 启动 QEMU → 运行推理。

成功输出 `ACT_INFER_OK`。

### 5. 清理

```bash
make docker-down    # 停止并删除容器
make clean          # 清理构建产物
```

## Make 目标一览

| 目标 | 执行位置 | 说明 |
|------|---------|------|
| `make export` | 主机 | PyTorch → ONNX 导出（dynamo） |
| `make verify` | 主机 | PyTorch vs ONNX 数值验证 |
| `make docker-up` | 主机 | 启动构建容器 |
| `make docker-shell` | 主机 | 进入容器交互式 shell |
| `make docker-down` | 主机 | 停止并删除容器 |
| `make host-test` | **容器内** | x86 构建 + ONNX 推理测试 |
| `make starry-test` | **容器内** | riscv64 交叉编译 + QEMU 测试 |
| `make test` | — | host-test + starry-test |
| `make clean` | 主机 | 清理产物（需 sudo 清除 root 属主文件） |

## 依赖

| 组件 | 位置 |
|------|------|
| Python 3.10+（torch、onnx、onnxruntime、onnxscript） | 主机 |
| Docker | 主机 |
| Rust（riscv64gc-unknown-linux-musl target） | 容器内置 |
| riscv64 musl 交叉工具链 | 容器内置 |
| QEMU riscv64 | 容器内置 |
| ONNX Runtime 1.26.0（libonnxruntime.so） | 容器内按需下载 |
