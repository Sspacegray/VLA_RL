# 验证记录

验证日期：2026-10-12。环境：macOS arm64、Python 3.12.13；torch 2.11.0、transformers 5.5.4、datasets 4.8.0、PyAV 13.1.0（本机既有测试环境，不等于 Linux 全量锁定依赖安装）。

## 已实际执行

| 检查 | 结果 | 含义 |
|---|---|---|
| 单目录合并 | 根目录只有一个 `src/lerobot`、`pyproject.toml` 和 `.git` | 以 0.6.2 为基础，移植 Piper 而非混装 0.4.2/0.6.2 |
| `python -m compileall -q src scripts benchmarks/robodojo` | 通过 | 语法，不是模型效果 |
| 真实 `PI05Policy` / `PI05RLTPolicy` import | 通过 | 使用 transformers 5.5.4，不依赖测试 stub |
| RLT + robots + motors + RoboDojo contract tests | **288 passed, 8 skipped** | CPU/SDK mock 数值与接口检查；不是动作成功率 |
| `uv lock --check` | 通过 | 根 pyproject 与 lock 文件一致 |
| Piper Stage2 模板 dry_run | 按预期拒绝 `/path/to/pi05/pretrained_model` | 无模型时拒绝继续，没有连接硬件 |
| Piper inference 模板 dry_run | 按预期拒绝 `/path/to/rlt_stage2/final_checkpoint.pt` | 无 Actor checkpoint 不启动 |
| RoboDojo 协议与结果汇总 | 合入上述测试 | 官方双臂 embodiment 不兼容时明确失败；无成绩 |

复现命令：

```bash
PYTHONPATH=src:scripts .venv/bin/python -m pytest -q tests/rlt tests/robots tests/motors \
  tests/test_robodojo_adapter.py tests/test_robodojo_summarize.py
```

跳过项为可选硬件/通信依赖路径。测试中的 SDK mock 不能证明 CAN/使能/机械安全。上游全测试套件没有全部执行；部分 rollout 合成数据测试依赖本机不具备的 `libsvtav1` 编码器。新增数据集测试显式使用 h264，未修改全局编码器默认行为。macOS 的 OpenCV/PyAV 存在系统动态库重复警告，生产部署推荐 Linux。

## 未完成，不应写成 PASS

- 本机未录制真实 Piper 演示，也未验证厂家关节零位、夹爪行程或安全 home。
- 无实际数据/权重，未完成 π0.5 微调、真实 embedding 预计算、在线 RL 或真机推理。
- 无 NVIDIA/Isaac Sim，因此无 RoboDojo 仿真 episode。
- 无双臂 Piper、官方云评测权限和双臂 checkpoint，因此无 RoboDojo 真机 episode。
- RoboDojo 当前 adapter 是**协议预检脚手架**，`_load_policy/get_action` 在兼容门控后仍明确未实现双臂推理。单臂 7D Actor 无法据此产生官方双臂 14D 动作。

要得到可比较的真实成绩，先按 `piper_workflow.md` 完成标定与单臂采训推；RoboDojo 需额外的双臂数据/模型/推理接线，在官方协议下完成相机/左右 state/夹爪映射验证后再进行 sim/real 两组正式实验。记录 checkpoint 哈希、seed、任务、layout、每回合成功/失败、接管次数与延迟；保持 VLA 与 RLT 对照条件一致，不补写缺失结果。

## 仓库交付

上游原始历史保存在 `../VLA_RL_upstream_backup/`；根仓库同步目标为 `https://github.com/Sspacegray/VLA_RL`。旧上游 GitHub workflows 归档到 `docs/upstream/workflows/`，避免 fork 自动尝试 Hugging Face 发布/文档上传；本项目仅启用 CPU contract CI。
