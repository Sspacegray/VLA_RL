# Piper 采集、训练、推理与前期标定

## 0. 先完成标定，不先跑策略

Piper 驱动的 `calibrate()` **只初始化当前位姿/平滑 home 并设置软件标志**，并没有测量或保存机械零位。因此 `lerobot-calibrate` 返回不等于机械标定完成。

建议把以下现场核验材料写入 `configs/local/calibration.md`（本机文件，不公开），必要的数值写到对应运行配置中：

| 项目 | 前期工作 | 对应文件/配置 |
|---|---|---|
| 机械零位/方向 | 按 AgileX 对应固件说明做关节零位校准，确认六关节正方向与限位；保存固件/SDK 版本、日期和厂家导出记录 | 厂家工具产物路径写入 `configs/local/calibration.md`；代码不会生成虚假零位 JSON |
| CAN | 确认 leader/follower 对应真实设备、1 Mbps，重启后仍正确 | `can_activate.sh`、`find_all_can_port.sh`；默认 `can_master`/`can_follower`，机器人配置 `can_name` |
| 夹爪 | 测量实际开闭行程、零偏、方向，核验米制反馈，最大行程以实物为准 | `PiperMotorsBus` 单位边界；`deployment.gripper.*`、home 第七维（m） |
| Home/关节安全 | 确认无碰撞起始位姿，逐轴低速试动，现场限位和每步最大变化需专门设定 | `deployment.home_position`（6 rad + 1 m）、`max_joint_step_rad`；初期 current/hold |
| 相机 | 固定设备路径、安装位置、旋转、分辨率、光照，采训推保持一致 | `configs/local/cameras.json`、`cameras`/`camera_map`；标定内参可放 `configs/local/camera_intrinsics/*.json` |
| 外参/手眼 | 若用 EEF、几何测量/对齐或 sim-to-real，测相机到基座/工具外参、TCP，保存方法和误差 | `configs/local/hand_eye.json`、`tcp.json`（约定记录文件，当前 joint 策略不自动读取） |
| FK/IK 模型 | EEF 必须验证 Piper URDF/MJCF、joint 顺序、轴符号、基座与 TCP，并接入 adapter | `robots/piper_follower/ik_adapter.py`；当前流程只使用 joint，未经验证禁止 EEF |
| 数据统计 | 用本机 Piper 演示重新计算，不使用 SO-101 的 6D stats | 数据集 `meta/info.json`、`meta/stats.json`，7D q01/q99 与特征名 |

图像 joint-space 策略不强制消费内参/外参；安装变化仍会使视觉分布变动。软件急停不代替硬件急停。首次测试必须有人监护。

## 1. 安装与硬件连通

按根 README 安装 Linux/Python 3.12 环境，先配置 CAN，再验证相机：

```bash
bash find_all_can_port.sh
# 替换 USB 地址，不能照抄另一工位
bash can_activate.sh can_master 1000000 "<leader USB 地址>"
bash can_activate.sh can_follower 1000000 "<follower USB 地址>"
ip -details link show can_follower
lerobot-find-cameras opencv
mkdir -p configs/local
```

配置相机到本机文件（例如两个固定设备 `top`、`wrist`）；`camera_map` 在预计算/在线脚本中必须与此一致。

```bash
export PIPER_CAMERAS='{"top":{"type":"opencv","index_or_path":"/dev/video_top","width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":"/dev/video_wrist","width":640,"height":480,"fps":30}}'
lerobot-teleoperate --robot.type=piper_follower --robot.cameras={} \
  --robot.deployment.start_from=current --robot.deployment.stop_at=hold \
  --teleop.type=piper_leader --display_data=false
```

## 2. 采集演示

```bash
export DATASET="$PWD/outputs/piper_demo"
export TASK='Pick up the object and place it in the box'
lerobot-record --robot.type=piper_follower --robot.cameras="$PIPER_CAMERAS" \
  --robot.deployment.start_from=current --robot.deployment.stop_at=hold \
  --teleop.type=piper_leader --display_data=true \
  --dataset.repo_id=local/piper_demo --dataset.root="$DATASET" \
  --dataset.single_task="$TASK" --dataset.num_episodes=30 \
  --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
  --dataset.fps=30 --dataset.push_to_hub=false
```

录制前先测 1 集并查看视频和关节量。Piper `record_action_from_follower=true` 延续上游行为：action 来自从臂反馈（EMA 后的实际状态），不是直接记录主臂目标；如果研究目标要求目标动作，应明确选择并全程保持同一种定义。

验收 `meta/info.json`：state/action 都是 7D，顺序为 `joint_1..joint_6,gripper`，关节 rad、夹爪 m。统计中必须有有效 7D 分位数；action、state 和 camera 特征需要与 checkpoint 配套。

## 3. π0.5 微调

仅复用适用的预训练 VLA 基座，不能直接部署 SO-101 Actor。用新的 Piper 数据和同一套源码重训：

```bash
lerobot-train --dataset.repo_id=local/piper_demo --dataset.root="$DATASET" \
  --dataset.video_backend=pyav --policy.type=pi05 \
  --policy.pretrained_path=/path/to/pi05_base \
  --policy.device=cuda --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true --policy.optimizer_lr=5e-4 \
  --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=outputs/pi05_piper --job_name=pi05_piper \
  --batch_size=1 --num_workers=4 --steps=30000 --save_freq=1000
```

先在一个**独立输出目录**跑 `--steps=3`，校验 finite loss、7D 特征、模型/归一化存档。正式产物使用 `checkpoints/<step>/pretrained_model/`（含 config、模型、processor），保持 tokenizer 本地可读。

## 4. RL Token Stage 1

```bash
export PI05=/path/to/pi05_piper/checkpoints/030000/pretrained_model
export TOKENIZER=/path/to/paligemma_tokenizer
python scripts/precompute_pi05_embeddings.py --pi05_path "$PI05" \
  --tokenizer_path "$TOKENIZER" --dataset_repo_id "$DATASET" \
  --state_dim=7 --robot_type=piper_follower \
  --camera_map '{"top":"observation.images.top","wrist":"observation.images.wrist"}' \
  --output_dir=outputs/piper_embeddings --batch_size=8 --num_workers=4 --device=cuda
python scripts/train_rlt_stage1_pi05.py --precomputed_path=outputs/piper_embeddings \
  --output_dir=outputs/rlt_stage1_piper --steps=30000 --batch_size=64 \
  --device=cuda --gpus=0 --use_amp
```

预计算产物为 mmap + meta；训练产物为 `best_checkpoint.pt`。先用 Stage1 `--dry_run` 做一次 finite forward/backward（需真实有效 embedding 缓存）。`z_rl`、Actor、统计必须追溯到同一个微调 VLA。

## 5. 在线 RL / 关键阶段

```bash
cp configs/piper_stage2_pi05.json configs/local/stage2.json
# 修改 pi05_path/rlt_checkpoint/tokenizer_path/stats_path/task/cameras/output_dir
python scripts/train_rlt_stage2_pi05.py --config configs/local/stage2.json
```

默认 `dry_run=true`：只预检，没有真机动作或学习成绩。确认配套后再改为 false，在急停保护下用 `--vla_only`，然后少量 Actor 回合，最后正式训练。建议 warmup 大于 15 集（小样本仅检查流程）。

Piper 不支持 SO-101 Feetech 主臂镜像/寄存器扭矩操作；不要套用度制 `max_relative_target`。Piper 限幅使用 `max_joint_step_rad`。回合 reset、HIL 起点和夹爪不能用同一个“大弧度阈值”盲目判定；任何异常须人工检查实物。首版推荐人工监护下单任务评估，关键阶段用 `c` 手动切换；分类器是可选训练步骤，原工具见 `CONFIG.zh-CN.md`，必须使用 Piper 7D state。

## 6. 推理与回合结果

```bash
cp configs/piper_infer_pi05.json configs/local/infer.json
# 填入同源路径与 resume_from=.../final_checkpoint.pt，保持 dry_run=true
python scripts/infer_rlt_stage2_pi05.py --config configs/local/infer.json
```

通过预检后再连接硬件：先 `vla_only`，后 Actor。开启 `inference_score` 做人工成功/失败统计，`record_dataset` 保存每执行步的新 observation，而不是重复 chunk 起始帧。对照实验固定初始布局、task、相机、频率、回合长度，分别测试 VLA baseline 和 RLT，记录失败、接管、停止，不只看挑选的视频。

## 7. Benchmark 验收边界

见 [RoboDojo 文档](benchmark_robodojo.md)。官方 Piper 是**双臂 14D joint**，本项目单臂 7D 不能直接上官方 benchmark。当前交付协议门控/安装/结果汇总，不交付伪造的双臂推理。下一阶段需要双臂硬件/数据/策略，明确两夹爪归一化及左右相机，再编写并验证 inference adapter，然后才能在 Isaac Sim 和官方真实场景对照测成绩。
