# Pi05_RLT (XPolicyLab adapter)

**Contributor:** VLA_RL | **Original code:** this repository
(`src/lerobot/policies/pi05_rlt`, `scripts/eval_rlt_pi05_so101.py`)

pi05_rlt is a π0.5 + RLT (residual-latent-token) policy stack from this
repo. Its only exercised eval path is SO-101 (6-D: 5 joints + gripper,
`scripts/eval_rlt_pi05_so101.py`); the repo's 7-D schema is the LOCAL
single-arm Piper joint action (6 joints rad + gripper m,
`docs/piper_real_robot_deployment.md`). **This repo ships no trained
checkpoints** (all paths in `configs/eval_pi05.json` are `/path/to/`
placeholders). This adapter is **integration scaffolding/preflight, not a
working policy adapter**: it serves the standard XPolicyLab websocket
protocol, and **fails closed on every RoboDojo embodiment** — see
*Limitations*.

## Installation

```bash
# from the VLA_RL repo root
bash benchmarks/robodojo/install_adapter.sh <ROBODOJO_ROOT> \
  --pi05-path /path/to/pi05/pretrained_model \
  --rlt-checkpoint /path/to/rlt_stage1/best_checkpoint.pt \
  --actor-checkpoint /path/to/rlt_stage2/final_checkpoint.pt
```

`install_adapter.sh` copies `model.py` / `deploy.py` / `__init__.py` /
`deploy.yml` into `<ROBODOJO_ROOT>/XPolicyLab/policy/Pi05_RLT/`, fills the
checkpoint paths, and reuses the reference `eval.sh` /
`setup_eval_policy_server.sh` / `setup_eval_env_client.sh` /
`install.sh` from `policy/demo_policy`. The policy conda env (arg 9 of
`eval.sh`) must be the VLA_RL environment (`uv sync` at the VLA_RL root, then
create/point at a conda env with the same packages — the stack needs
`lerobot` from this repo importable, which `serve_policy.sh` does via
PYTHONPATH).

## Data Processing

Not supported. pi05_rlt trains from the VLA_RL warm-up/critical datasets
(`scripts/train_rlt_stage1_pi05.py`, `scripts/train_rlt_stage2_pi05.py`),
not from XPolicyLab RoboDojo trajectories.

## Training

Not supported through this adapter; train with the VLA_RL scripts above.

## Evaluation

Debug loop (no simulator):

```bash
export EVAL_ENV_TYPE=debug
cd <ROBODOJO_ROOT>/XPolicyLab/policy/Pi05_RLT
bash eval.sh RoboDojo stack_bowls pi05_rlt arx_x5 joint 0 0 0 <VLA_RL_env> base
```

The server will start, load the adapter, and immediately raise
`EmbodimentMismatchError` — that is the intended fail-closed behavior for
bimanual `arx_x5` (14-D) against a single-arm 7-D schema, not a bug.

Simulation (needs Isaac Sim, Linux + RTX GPU):

```bash
bash <ROBODOJO_ROOT>/scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi05_RLT \
  --task stack_bowls --ckpt pi05_rlt --policy-env <VLA_RL_env>
```

## Configuration

`deploy.yml` keys: `pi05_path`, `rlt_checkpoint`, `actor_checkpoint`,
`stats_path` (optional stats.json override), `actor_execution_steps`,
`rtc_execution_horizon`, plus the XPolicyLab timeout keys (the 3B backbone
cold load justifies `request_timeout_s: 600`).

## Limitations

- **Fail closed on all RoboDojo embodiments.** RoboDojo's robots
  (`arx_x5`, `piper`, `piper_x`) are bimanual
  (`arm_dim=[6,6], ee_dim=[1,1]`, 14-D). The local single-arm Piper schema
  is 7-D (6 joints rad + 1 gripper m) — RoboDojo's `piper`/`piper_x` are
  dual Piper, a different embodiment. (SO-101 is 6-D: 5 joints + gripper,
  and is not a RoboDojo embodiment at all.)
  `model.py::assert_compatible` raises `EmbodimentMismatchError` before
  loading torch; no padded/mirrored pseudo-conversion is attempted.
- **No trained weights, by design.** This repo contains no trained
  checkpoints; `_load_policy` and `get_action` raise `NotImplementedError`.
  The only hypothetical pass case is an unregistered `single_piper`
  embodiment with `action_type: joint`. When real checkpoints for a matching
  embodiment exist, implement `_load_policy` + `get_action` following
  `scripts/eval_rlt_pi05_so101.py`.
- No leaderboard claim: this adapter has never produced a RoboDojo score.
