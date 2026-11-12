# DexVerse + LeRobot PI0.5 策略扩展

## 目标

本分支在 Tracking Streamer 遥操数采链路的基础上，增加从示范数据到 LeRobot Dataset、PI0.5 LoRA 微调以及闭环仿真评估的完整流程：

```text
遥操 PKL
  -> Isaac Sim 重放并生成 demo HDF5
  -> LeRobot Dataset v3（Parquet + H.264 视频）
  -> PI0.5 LoRA 微调
  -> Unix socket 推理服务
  -> DexVerse 闭环评估
  -> 成功率、轨迹和 MP4
```

PI0.5 推理被拆成两个进程：Isaac Sim 进程负责环境、相机和评估循环；LeRobot 环境中的独立服务负责加载模型和推理。两者通过本机 Unix socket 通信，不需要把 LeRobot 依赖安装到 DexVerse 的 Isaac Sim 环境中。

## 新增环境配置

### `DexVerse/lerobot.yaml`

提供独立的 `lerobot` Conda 环境配置，包含 LeRobot、PI0.5 所需的 PyTorch、Transformers、PEFT、HDF5、视频编码和数据处理依赖。该环境与运行 Isaac Sim 的 `dexverse` 环境分开使用。

注意：该 YAML 包含当前机器/时间生成的精确版本和 CUDA 依赖，迁移到其他机器时应根据 GPU、CUDA 和 LeRobot 版本重新求解环境。

## HDF5 转 LeRobot Dataset

### `scripts/demo_tools/dexverse_h5_to_lerobot.py`

将 `create_demo_files_sequential.sh` 生成的 DexVerse demo HDF5 转换为 LeRobot Dataset v3：

- 读取每个 HDF5 文件中的 `data/demo_*` episode。
- 从 `obs/proprio/joint_pos` 保留最近一帧的 28 维 Shadow Hand 状态。
- 写入 28 维 `action`。
- 将四路 RGB 写为 `observation.images.rgb_image`、`left_rgb_image`、`right_rgb_image` 和 `wrist_rgb_image`。
- 保存任务文本、episode/frame 信息以及视频元数据。
- 使用 60 FPS 和 H.264 视频编码；可通过参数覆盖 FPS、任务名和输出目录。


### `scripts/demo_tools/dexverse_h5_to_lerobot.sh`

提供上述转换的快捷启动器，默认使用 `lerobot` 环境和 60 FPS。脚本中的 Conda 路径、数据路径以及 Python 文件路径是机器相关配置，提交前应改为可配置参数；直接使用时请确认这些路径与本机一致。

## PI0.5 微调

### `scripts/train/finetune_pi05.sh`

使用 `lerobot-train` 对 `lerobot/pi05_base` 进行 LoRA 微调：

- 数据集：`Dexverse-PickCube-v0`。
- 策略：`pi05`，`bfloat16`，CUDA。
- LoRA：rank `16`，alpha `16`。
- 训练步数：`20000`，每 `5000` 步保存一次。
- 观测历史：`n_obs_steps=3`。
- 开启 gradient checkpointing，冻结视觉编码器。
- 输出：`hf_ckpts/finetune/pi05/Dexverse-PickCube-v0-pi05`。

启动前需要确认 `lerobot_datasets/Dexverse-PickCube-v0` 已生成，并修改脚本中的 Conda、仓库和输出路径以适配当前机器。

## PI0.5 推理服务

### `scripts/lerobot_pi05_server.py`

在 `lerobot` 环境中加载 PI0.5 的 PEFT/LoRA `pretrained_model`，并提供本机 Unix socket 服务：

- `reset`：重置策略和预处理器状态。
- `predict_chunk`：接收 28 维状态和四路 HWC RGB，返回一段动作 chunk。
- `close`：关闭服务。

输入先经过 LeRobot 的预处理器，图像由 HWC 转为视觉编码器使用的 CHW；模型输出再经过后处理器还原到数据集动作尺度。Socket 文件默认是 `/tmp/dexverse_pi05.sock`，权限为 `0600`，仅当前用户可访问。

### `scripts/lerobot_pi05_server.sh`

激活 `lerobot` 环境并启动推理服务，默认加载 `020000/pretrained_model` 检查点。可通过 `PI05_DEVICE` 选择推理设备。检查点、数据集和 Conda 路径目前是机器相关的，应在贡献前改为参数或环境变量。

## DexVerse 评估

### `scripts/lerobot_pi05_client.py`

作为 `eval.py` 的外部策略模块运行，负责：

- 将 DexVerse 当前 28 维机器人状态和四路相机帧转换成 LeRobot 字段名。
- 通过 Unix socket 向服务端请求动作 chunk。
- 缓存 chunk，并在每个仿真步取出一个动作。
- 推理等待期间定期调用 keep-alive 回调，避免 Isaac Sim 界面失去响应。

### `scripts/eval.py`

提供通用的单环境策略评估入口，支持 `module`、`checkpoint`、`zero` 和 `random` 策略模式，并负责：

- 顺序运行指定数量的 episode。
- 调用任务的 `success` 条件统计成功率。
- 保存每个 episode 的多相机 MP4。
- 写出评估汇总和逐 episode 结果。
- 通过 `--eval_render_mode` 选择 `gui`、`rgb` 或 `headless`。
- 通过 YAML 应用环境和事件随机化。

### `scripts/eval_domain_randomization.yaml`

提供评估随机化模板，包括物体位姿/速度、机器人关节、背景光照、曝光和色温等 reset event。当前示例默认保持稳定基线；需要随机化时可在 YAML 中启用对应事件。

### `scripts/pi05_eval.sh`

提供 PI0.5 闭环评估启动器：加载 `lerobot_pi05_client.py`，使用 `3view_rgb` 观测，默认运行 30 个 episode，并将结果写入 `outputs/eval`。典型流程需要两个终端：

```bash
# 终端 1：启动 LeRobot 推理服务
PI05_DEVICE=cuda bash scripts/lerobot_pi05_server.sh

# 终端 2：启动 Isaac Sim 评估
bash scripts/pi05_eval.sh
```

评估脚本中的 `CHECKPOINT`、任务名、socket、输出目录和设备均可按机器修改；默认路径指向本项目当前的 `/home/dehand/dex_eval` 布局。
