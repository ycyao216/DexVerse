# 基于 Tracking Streamer + avp_stream 的 DexVerse 遥操数采扩展

## 目标

使用 Vision Pro 上的 **Tracking Streamer** 应用和 Python 包 **`avp_stream`**，将手部追踪数据直接接入 DexVerse，驱动 Shadow Hand 并使用 `record_demos.py` 保存成功轨迹。本文所称“Tracking Streamer 方案”特指这条直连链路，以区别于同样使用 Vision Pro 硬件的 CloudXR 方案。

该方案不经过 Docker、CloudXR 或 OpenXR Runtime：

```text
Vision Pro Tracking Streamer
  -> 局域网 avp_stream
  -> VisionProDevice
  -> DexVerse 手部重定向
  -> Shadow Hand
  -> record_demos.py
  -> 轨迹 .pkl
```

## 环境

将`avp_stream 2.51` 及相关依赖项安装在现有 `dexverse` Conda 环境中。当前兼容的关键版本为：
```text
numpy 1.26.0
opencv-python 4.11.0
torch 2.7.0+cu128
```
不要将 NumPy 升级到 2.x，具体的环境配置见DexVerse/dexverse.yaml。

## 新增文件

### `DexVerse/source/dexverse/dexverse/devices/visionpro_device.py`

新增 DexVerse 设备 `VisionProDevice`，主要负责：

- 通过 `VisionProStreamer` 读取左右手 25 个关节的 `4x4` 变换矩阵。
- 将 Vision Pro 关节映射为 DexVerse/OpenXR 兼容的关节名称，并补充 `palm`。
- 将矩阵转换为 `[x, y, z, qw, qx, qy, qz]` 姿态。
- 默认绕世界 `Z` 轴旋转 `-90` 度，使真实手方向与 Isaac Sim 场景一致。
- 根据手部关节位置构建 Shadow Hand 所需的手掌坐标系。
- 数据短暂丢帧时复用上一帧有效姿态。
- 提供 `S/P/R/Q` 键盘控制。

### `DexVerse/scripts/record_demos_visionpro.sh`

- Vision Pro 数采入口，负责激活环境并调用。
- 支持通过环境变量修改 IP、任务和采集数量。

### `DexVerse/scripts/device_tools/visionpro_hand_diagnostic.py`

- 不启动 Isaac Sim，单独采集张手和握拳数据.
- 用于在出现bug时检查问题来自 Vision Pro 原始数据还是 DexVerse 重定向。

### `DexVerse/scripts/demo_tools/create_demo_files_sequential.sh`
- 数据转换，创建demo，详情见下方**数据格式**。

## 修改的 DexVerse 源码

- `devices/__init__.py`：导出 `VisionProDevice` 和 `VisionProDeviceCfg`。
- `tasks/config/floating_teleop.py`：注册名为 `visionpro` 的遥操设备，并连接现有 relative/absolute retargeter。
- `devices/retargeters/simple_relative_retargeting.py`：允许 Vision Pro 提供独立的手掌标准坐标系，解决张手时 Shadow Hand 自动蜷缩的问题；同时避免私有坐标矩阵导致点云可视化崩溃。
- `scripts/record_demos.py`：让 Vision Pro 自动启用手部重定向依赖，添加 `Quit` 控制、用于正常退出整个数采程序，保存已完成轨迹并丢弃当前未完成的轨迹。

IsaacLab 源码没有修改，原来的 CloudXR/OpenXR 设备也仍然保留。

## 控制方法

先在 Vision Pro 的 Tracking Streamer 中开始发送数据，再运行：

```bash
cd /path/to/DexVerse
./scripts/record_demos_visionpro.sh
```

Isaac Sim 打开后，将焦点放在主窗口：

```text
S  校准当前手腕姿态并开始录制
P  暂停录制，不完成当前轨迹
R  放弃当前未成功轨迹并重置环境
Q  正常结束本次数采
```

当具体任务的成功条件连续满足 `NUM_SUCCESS_STEPS` 次数后，当前 episode 自动完成并保存。达到 `NUM_DEMOS` 后程序自动退出；设置 `NUM_DEMOS=0` 可持续采集直到手动按 `Q`结束。

## 数据格式

一次启动生成一个带时间戳的 `.pkl` 文件，多次成功操作保存为其中的多个 episode。每个 episode 包含：

```text
initial_state  初始场景状态
actions        T x 28 动作（手腕 6 维 + 手指 22 维）
states         T+1 个逐步场景状态
success        成功标记
num_steps      轨迹步数
```

默认 pkl 存储目录为：

```text
DexVerse/visionpro_test/<task>/...
```

使用 `scripts/demo_tools/create_demo_files_sequential.sh` 将指定任务目录中的所有 pkl 逐个转换为 demo（HDF5 + MP4），每个 episode 对应一个 demo：

```bash
cd /path/to/DexVerse
TASK_NAME=Dexverse-PickCube-v0 ./scripts/demo_tools/create_demo_files_sequential.sh
```

该脚本会为每个 pkl 启动一个独立的 Isaac Sim 进程，按记录的初始场景状态和动作顺序重放各个 episode，同时采集 `3view_rgb` 三视角观测。输出文件为：

```text
DexVerse/demo/visionpro_test/<task>/<pkl文件名>.3view_rgb.seq.demo.h5
```

hdf5 使用类似文件系统的 Group/Dataset 层次结构，支持压缩、局部读取和跨语言处理。每个转换结果的核心结构为：

```text
data/
├── demo_0/
│   ├── actions
│   ├── source_actions
│   ├── obs/          当前帧观测，包含三视角 RGB
│   ├── next_obs/     下一帧观测
│   ├── initial_obs/  episode 初始观测
│   ├── final_obs/    episode 结束观测
│   └── terminations/ 终止条件
└── demo_1/ ...
```

转换完成后，脚本还会为各 episode 生成三视角合成 MP4，保存到：

```text
DexVerse/demo/visionpro_test/composite_videos/<task>/<pkl文件名>/<各demo的mp4文件>
```

## 与 CloudXR 的区别

Tracking Streamer 方案最重要的实际区别是：**不需要 Apple Silicon Mac 和 Xcode，也不需要自行编译、签名并向 Vision Pro 安装官方的 `Isaac XR Teleop Sample Client App`**。本方案直接使用 Vision Pro App Store 中可直接安装的应用 Tracking Streamer，通过局域网发送手部骨架数据。

官方 CloudXR + Vision Pro 流程需要在 Mac 上克隆 `isaac-xr-teleop-sample-client-apple` 仓库，使用匹配版本的 Xcode 构建客户端，再将它部署到 Vision Pro；没有 Mac 时很难完成这一步。

此外，本方案还不需要：

- 启动 `cloudxr-runtime` Docker 容器。
- 设置 `XDG_RUNTIME_DIR` 或 `XR_RUNTIME_JSON`。
- 创建 OpenXR IPC socket。
- 使用 CloudXR 网络端口或 `--xr` 模式。

缺点是 Tracking Streamer 方案只把头部/手部追踪数据传给 DexVerse，仿真画面仍主要在 Ubuntu 上的 Isaac sim 上显示；无法像 CloudXR 那样把双目沉浸式仿真画面实时串流回 Vision Pro。
