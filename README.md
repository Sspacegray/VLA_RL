# VLA_RL

Piper 六轴机械臂的 **采集 → π0.5 微调 → RL Token → 在线 RL → 推理**，统一为一个源码目录、一个 Python 环境。

## 当前状态

- 使用 RL-Token-Pi05-open 的 LeRobot **0.6.2** 作为唯一基础，合入 Piper SDK/CAN 驱动和部署保护；没有两个同名 `lerobot` 包。
- Piper joint 状态/动作均为 **7 维**：`joint_1..joint_6`（rad）+ `gripper`（m）。SO-101 仍为 6 维，不能混用其权重、统计或配置。
- [完整采训推步骤与标定清单](docs/piper_workflow.md)
- [Piper 部署保护](docs/piper_real_robot_deployment.md)
- [RoboDojo 仿真/真机评测与阻塞项](docs/benchmark_robodojo.md)
- [验证记录](docs/validation.md)

**没有宣称完成真机或 RoboDojo 效果测试。** 当前机器没有 NVIDIA GPU、Piper 硬件、训练数据和模型权重。RoboDojo 当前为双臂协议，本项目单臂 Piper 7D 不可直接提交；评测适配器会明确拒绝不兼容的 embodiment，不能用补零伪造成绩。

## 安装（训练/真机推荐 Linux + CUDA）

```bash
git clone https://github.com/Sspacegray/VLA_RL.git
cd VLA_RL
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[piper,pi,test]'
```

相机、CAN、关节零位、夹爪行程与安全 home 必须先确认。首次上真机使用 `start_from=current`、`stop_at=hold`，不要把软件 `calibrate()` 当作厂家机械零位标定。

## 配置与检查

```bash
mkdir -p configs/local
cp configs/piper_stage2_pi05.json configs/local/stage2.json
cp configs/piper_infer_pi05.json configs/local/infer.json
# 替换模型/统计/相机路径，保持 dry_run=true
python scripts/train_rlt_stage2_pi05.py --config configs/local/stage2.json
python scripts/infer_rlt_stage2_pi05.py --config configs/local/infer.json
```

模板路径是占位符，未填写时检查应失败；这不是一次训练或推理成绩。

```bash
PYTHONPATH=src:scripts pytest -q tests/rlt tests/robots/test_piper_deployment.py \
  tests/test_robodojo_adapter.py tests/test_robodojo_summarize.py
```

## 结构

```text
src/lerobot/                       唯一 LeRobot 源码（0.6.2）
  policies/pi05_rlt/               RL Token / Actor / Critic
  robots/piper_follower/           CAN follower 与部署保护
  teleoperators/piper_leader/      Piper 遥操作
scripts/                          RLT 预计算、训练、推理
configs/piper_*.json               Piper 安全默认模板（dry_run）
benchmarks/robodojo/               官方协议校验、启动工具、结果汇总
configs/local/                    本机配置（不提交）
outputs/                          数据/权重/日志（不提交）
docs/                             操作步骤、标定与验证记录
```

上游原始仓库（含历史）保留在本机邻接目录 `../VLA_RL_upstream_backup/`，不放进 GitHub，不形成嵌套 Git 仓库。上游说明留在 `docs/upstream/`，其相对链接及安装命令仅作为历史参考。

## 许可证与来源

本项目按 Apache-2.0 发布；保留 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

- [Hugging Face LeRobot](https://github.com/huggingface/lerobot)
- [Piper LeRobot](https://github.com/jokeru8/piper_lerobot)，基线 `aa393f5`
- [RL-Token-Pi05-open](https://github.com/Zhengsw03/RL-Token-Pi05-open)，基线 `cccf949`
- RoboDojo / XPolicyLab 保持外部依赖，不把其代码或成绩冒充本项目实现。
