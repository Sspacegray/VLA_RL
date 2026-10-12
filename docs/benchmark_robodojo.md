# RoboDojo benchmark integration (pi05_rlt / Piper)

Status: **integration scaffolding/preflight only — NOT a working policy
adapter**; loader/get_action raise `NotImplementedError`, this repo has no
trained checkpoints, and every real RoboDojo run is blocked locally (see
[Blockers](#blockers)). No RoboDojo score has been produced or is claimed
anywhere in this repo.

What this integration is:

- `benchmarks/robodojo/xpolicylab_adapter/pi05_rlt/` — a standard
  [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) policy adapter for
  the VLA_RL `pi05_rlt` stack, copied into an official RoboDojo clone by
  `install_adapter.sh`. It speaks the official websocket policy-server
  protocol and **fails closed** on every current RoboDojo embodiment.
- `benchmarks/robodojo/summarize_metrics.py` — deterministic local view of
  `_result.json` outputs (success rate/score + optional latency).
- `tests/test_robodojo_adapter.py`, `tests/test_robodojo_summarize.py`.

What it is **not**: a re-implementation or vendored copy of RoboDojo; a
simulator; a claim on the leaderboard.

Evidence base: RoboDojo @ `cb2a37d` (2026-10-08) and XPolicyLab @ `f6e309e`
(2026-10-11), cloned to `/tmp/RoboDojo` and `/tmp/XPolicyLab`; the
`env_cfg/robot/_robot_info.json`, `scripts/robodojo.sh`,
`setup_policy_server.py`, `utils/process_data.py`, `utils/debug_env_client.py`
files quoted below were read directly from those clones. RoboDojo paper:
arXiv:2607.04434; XPolicyLab paper: arXiv:2608.09892.

---

## 1. What RoboDojo is

Isaac-Sim-based manipulation benchmark, evaluated through XPolicyLab's
client–server protocol:

- **42 simulation tasks** across 5 dimensions (generalization, memory,
  precision, long-horizon, open); `_random` variants are paired-layout
  generalization probes.
- **18 real-robot tasks** on three embodiments: **Piper X, Piper, ARX X5** —
  all **bimanual**.
- Entry point: `bash scripts/robodojo.sh <doctor|eval|server|client|smoke|
  benchmark|dimensions|summarize>`.
- Result layout (written by the env client):
  `eval_result/RoboDojo/<task>/<policy>/<embodiment>/<seed>_ckpt=.../<timestamp>/_result.json`
  with `details: {layout_id: {layout_id, success, score}}`.
- Official aggregation (`robodojo.sh summarize`): 50 episodes standalone, or
  25+25 paired with the `_random` twin; seeds 0/1/2; 126 canonical cells.

## 2. Protocol (policy side — what the adapter implements)

The policy runs as a websocket server; the benchmark is the client.

1. `eval.sh <bench> <task> <ckpt> <env_cfg_type> <action_type> <seed>
   <policy_gpu> <env_gpu> <policy_env> <eval_env_conda_env>` starts
   `setup_eval_policy_server.sh` → `XPolicyLab/setup_policy_server.py`
   → imports `XPolicyLab.policy.<policy_name>.model.Model` and instantiates
   it with `deploy.yml` merged with the per-run overrides.
2. Server frames: `HELLO → PREPARE_CASE → RESET → CALL(update_obs) →
   CALL(get_action) → ... → TRIAL_END → HEARTBEAT`. The server decodes
   `obs["vision"][<cam>]["color"]` into RGB arrays **before** dispatch, so
   `model.py` never decodes images (enforced by test + CONTRIBUTING greps).
3. Observation keys (`utils/debug_env_client.py`): `obs["state"]` with
   `left_/right_arm_joint_state`, `ee_joint_state`, `ee_pose`, `tcp_pose`,
   `delta_ee_pose`; `obs["instruction"]`. Actions are dicts with the same
   keys; `get_robot_action_dim_info(env_cfg_type)` reads
   `env_cfg/robot/_robot_info.json`.
4. `get_action()` returns **one chunk**: `list[dict[str, np.ndarray]]`; the
   env steps through it until episode end. Per-call `latency_ms` is attached
   by the server to every result payload.
5. `EVAL_ENV_TYPE=debug|sim|real_world` selects the client: synthetic obs
   (no simulator), Isaac Sim, or the real robot.
6. `Model.reset()` is called per episode (TRIAL_END); a policy needing a
   first observation waits for the next `update_obs`.

Verified end-to-end on this machine (see § Debug loop below): the official
`setup_policy_server.py` loads our installed adapter and fails closed at the
embodiment gate.

## 3. Why the adapter fails closed (single-arm 7-D Piper vs bimanual)

What exists in this repo (no trained checkpoints — `configs/eval_pi05.json`
holds `/path/to/` placeholders only):

- **SO-101 eval script** (`scripts/eval_rlt_pi05_so101.py`): 6-D — **5 joints
  + 1 gripper** (`action_dim = 6`, per
  `src/lerobot/policies/pi05_rlt/configuration_pi05_rlt.py`). SO-101 is
  **6-D, not 7-D**, and is not a RoboDojo embodiment at all.
- **Local single-arm Piper schema** (`docs/piper_real_robot_deployment.md`):
  7-D joint action = **6 joints (rad) + 1 gripper (m)**, single arm, direct
  CAN — it does not use the XPolicyLab protocol.

RoboDojo's registry (`env_cfg/robot/_robot_info.json`, read live from the
official clone):

```json
{ "dual_x5": { "arm_dim": [6, 6], "ee_dim": [1, 1] } }
```

`arx_x5`, `piper`, `piper_x` all resolve to `dual_x5`-style bimanual robots:
a 14-D joint action or 2×(7-D ee pose + 1-D gripper). There is no lossless
7-D → 14-D mapping: padding or mirroring would invent actions the policy
never learned and silently corrupt any leaderboard number. So
`model.py::assert_compatible(env_cfg_type, action_type)` raises
`EmbodimentMismatchError` **before torch is imported**, with an actionable
message. Unknown `env_cfg_type` and missing `env_cfg_type` also fail closed
(there is deliberately **no `so101` entry**: SO-101 is 6-D and must not be
smuggled through the gate as a 7-D single arm).

Additionally `_load_policy()` and `get_action()` raise
`NotImplementedError`: **this integration is scaffolding/preflight, not a
working policy adapter** — no checkpoint has ever been trained for any
RoboDojo embodiment here. `update_obs` / `update_obs_batch` /
`get_action_batch` / `reset` plumbing is implemented and unit-tested; when
real checkpoints for a matching embodiment exist, implement `_load_policy`
+ `get_action` following `scripts/eval_rlt_pi05_so101.py` and drop the
guard table entry.

The one hypothetical pass case (`single_piper`: 1 arm, 6+1 = 7-D joint
action — the local Piper schema, **not registered in RoboDojo today**) is
kept so the gate is a real dimension check, not a hardcoded `raise`; it
requires `action_type: joint` (no EEF front end exists).

## 4. Local Piper protocol ≠ RoboDojo leaderboard

Two unrelated Piper stacks; do not conflate:

| | Local Piper deployment (this repo) | RoboDojo Piper tasks |
| --- | --- | --- |
| Transport | direct CAN via `lerobot-robot-client` + `piper_follower` (see `docs/piper_real_robot_deployment.md`) | XPolicyLab websocket policy server + cloud eval client |
| Embodiment | single Piper arm, 7-D joint (6 rad + gripper m) or 8-D EEF mode | `piper`/`piper_x` = **dual** Piper, 14-D joint / ee |
| Data contract | `observation.state`/`action` in LeRobot datasets, quantile stats | XPolicyLab `unpack_robot_state` dicts, `decode_image_bit` |
| Scores | none (deployment only) | leaderboard entries via cloud evaluation |

Running the local Piper stack proves nothing about RoboDojo, and vice versa.

## 5. Launch commands

Prereqs for anything beyond the unit tests: Linux + NVIDIA GPU (Isaac Sim),
a RoboDojo conda env, and a VLA_RL env. See [Blockers](#blockers) for why
none of these run on this machine.

```bash
# 0) one-time: official clones (not vendored here)
git clone https://github.com/RoboDojo-Benchmark/RoboDojo /tmp/RoboDojo
git -C /tmp/RoboDojo submodule update --init XPolicyLab

# 1) install the adapter + YOUR checkpoints (script refuses missing paths)
cd /Users/wubinxin/work/VLA_RL   # repo root
bash benchmarks/robodojo/install_adapter.sh /tmp/RoboDojo \
  --pi05-path /path/to/pi05/pretrained_model \
  --rlt-checkpoint /path/to/rlt_stage1/best_checkpoint.pt \
  --actor-checkpoint /path/to/rlt_stage2/final_checkpoint.pt

# 2) debug closed loop — no simulator, synthetic obs; expected outcome for
#    RoboDojo embodiments is the fail-closed error (§3), which proves the
#    server loads and the gate fires:
export EVAL_ENV_TYPE=debug
cd /tmp/RoboDojo/XPolicyLab/policy/Pi05_RLT
bash eval.sh RoboDojo stack_bowls pi05_rlt arx_x5 joint 0 0 0 <VLA_RL_env> base

# 3) simulation (Isaac Sim machine)
bash /tmp/RoboDojo/scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/Pi05_RLT --task stack_bowls \
  --ckpt pi05_rlt --policy-env <VLA_RL_env> --env-cfg arx_x5 --action-type joint
# batch across the benchmark: `robodojo.sh smoke|benchmark --dimension ...`

# 4) split-machine / own-shell server
bash benchmarks/robodojo/serve_policy.sh /tmp/RoboDojo <VLA_RL_env> <port> [host]

# 5) summarize local results (official aggregator: robodojo.sh summarize)
python3 benchmarks/robodojo/summarize_metrics.py eval_result/RoboDojo
```

**Official cloud / real eval** (leaderboard): open a PR to XPolicyLab adding
`policy/Pi05_RLT/` per `CONTRIBUTING.md` (adapter + README + checkpoint
download script so results are reproducible), then apply through RoboDojo's
cloud evaluation pipeline (robodojo-benchmark.com → Submission; the website's
form pairs with the PR). Anti-cheating verification reruns your checkpoint.
Real-robot tasks (Piper X / Piper / ARX X5, 18 tasks) go through the same
cloud pipeline — they need the physical dual-arm robot and are **not**
divided into the five dimensions.

## 6. Reset & calibration semantics

- `RESET`/`TRIAL_END` → `Model.reset()`; our adapter clears the observation
  cache and chunk state. Local Piper calibration (joint zeroing, EMA seeds,
  home interpolation — `docs/piper_real_robot_deployment.md`) is part of the
  local deployment stack and **does not exist** in the benchmark protocol;
  RoboDojo's sim resets layouts/scenes itself.
- `ee_joint_state` normalization mismatch is another reason no conversion is
  attempted: RoboDojo grippers are normalized dims against the local Piper
  gripper in meters.

## 7. Metric summarizer

`benchmarks/robodojo/summarize_metrics.py` walks
`<task>/<policy>/<embodiment>/<run>/<timestamp>/_result.json`, uses the
latest timestamp per run, and prints a sorted markdown table (or `--json`):
episodes, successes, success rate %, score %, and — only if a sibling
`_latency.json` with `calls_ms` exists — p50/p95/mean ms. Runs without
`_result.json` are reported as missing, never as 0%. Deterministic output
(byte-identical across runs) is pinned by test. For leaderboard-shaped
aggregation (25+25 pairing, 3 seeds, 126 cells) use the official
`robodojo.sh summarize`.

## 8. Tests

```bash
PYTHONPATH=scripts:. uv run --with pytest --with pyyaml pytest \
  tests/test_robodojo_adapter.py tests/test_robodojo_summarize.py -q
```

33 tests: fail-closed gate (all 3 embodiments × joint/ee, missing/unknown
env_cfg, registry unavailable, `so101` rejected, `ee` rejected), adapter
plumbing, deploy.yml standard keys, install-script checkpoint validation,
no-image-decode static check, summarizer determinism/missing-data/latency
merge.

## 9. Blockers (why no real run happened here)

1. **macOS CPU-only laptop.** Isaac Sim/IsaacLab need Linux + NVIDIA GPU;
   RoboDojo sim tasks cannot start. (The `EVAL_ENV_TYPE=debug` loop is the
   only locally runnable path, and its outcome for RoboDojo embodiments is
   the intentional fail-closed error.)
2. **No robot.** The 18 real tasks need a physical bimanual Piper X / Piper /
   ARX X5 + cloud pipeline access.
3. **No trained checkpoints in this repo.** `configs/eval_pi05.json` still
   has `/path/to/...` placeholders. Nothing here is trained for any RoboDojo
   embodiment: the SO-101 stack is 6-D single-arm (5 joints + gripper), the
   local Piper schema is 7-D single-arm, and RoboDojo's embodiments are all
   bimanual 14-D (§3). Even with GPUs, a leaderboard entry first needs a
   checkpoint actually trained for a matching embodiment.

Consequence: this integration ships the adapter, the summarizer, tests, and
documentation — and zero benchmark numbers, by design.
