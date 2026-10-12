# Piper (7D) RLT Support Report

Date: 2026-10-12
Scope: RLT (RL-Token) π0.5 pipeline — collection → stage1 → stage2 → inference — now runs on the Piper arm (`robot_type=piper`) in addition to SO-101.

## What changed

### `scripts/rlt_core.py` (shared runtime core)
- `PIPER_JOINT_NAMES = ("joint_1".."joint_6", "gripper")` — 7D; joints in **radians**, gripper in **meters** (native Piper units), same `"<name>.pos"` dict convention as SO-101.
- `RobotProfile` dataclass + `ROBOT_PROFILES` + `robot_profile(robot_type)`: per-robot `joint_names`, `state_dim`/`action_dim`, and capability flags (`uses_serial_port`, `supports_leader_mirror`, `supports_torque_control`, `policy_action_method`). Unknown types fail closed.
- `action_order_indices(..., allowed=None)` / `reorder_action_tensor(..., allowed=None)`: the allowed joint set is now a parameter; the default remains SO-101 (backward compatible).
- `_validate_quantiles(..., expected_dim=6)` and `load_normalization_stats(..., action_dim=6)`: 7D quantile stats validate on Piper; 6D default unchanged.
- `make_robot_pair(robot_type, ...)`: lazy-import hardware factory. `piper` → `PIPERFollower`/`PIPERLeader` (CAN bus; no ports; `max_relative_target` rejected with a clear error; `max_joint_step_rad` → `deployment` config). `so101` → exact legacy `SO101Follower`/`SO101Leader` construction.
- `follower_send_action(follower, action)`: prefers `send_policy_action` when the robot provides it (Piper: deployment step limit + gripper filter); SO-101 keeps `send_action` unchanged.

### `scripts/train_rlt_stage2_pi05.py` / `scripts/infer_rlt_stage2_pi05.py`
- New `--robot_type {so101,piper}` (default so101) and `--max_joint_step_rad` (Piper per-step joint delta limit, radians, applied inside `send_policy_action`).
- `PI05RLTConfig(state_dim, action_dim)` now comes from the profile (7 on Piper, ≤32 max so no policy change needed).
- `JOINT_NAMES` (action/state tensor mapping) and `TeleopManager.JOINT_NAMES` come from the profile; SO-101 default byte-identical.
- `send_validated_action` routes through `follower_send_action` → `send_policy_action` on Piper.
- **Leader mirror**: thread start, `mirror_follower_to_leader`, `enable_torque`, `bus.write("Torque_Limit", ...)`, `send_feedback` all live in the mirror loop — the loop is simply never started on Piper (`supports_leader_mirror=False`); publish calls are no-ops. Cleanup `leader.disable_torque()` guarded by `hasattr` (PiperLeader has no such method).
- Stats validated at the profile dim; camera configs pass through `rotation` (needed by the Piper camera setups); rollout recorder `robot_type` becomes `piper_follower` and state/action features follow `state_names` (7D).
- `--max_relative_target` is rejected on Piper at startup (Feetech degree clamp, not applicable); serial ports not required on Piper (CAN).
- Stage2 `runtime_provenance` now records `robot_type` and `joint_names`.

### Supporting scripts
- `export_rlt_pretrained.py`: `action_feature_names` / `state_names` / `action_names` / `robot_type` derive from stage2 `runtime_provenance` (falls back to SO-101 for legacy checkpoints); mismatches with `action_dim` fail closed.
- `precompute_pi05_embeddings.py`: `V3EmbeddingWriter` takes `state_dim` + `robot_type` (new `--state_dim`, `--robot_type` flags); no more hardcoded 6D validation.
- `verify_rlt_checkpoints.py`: stage2 dims inferred from provenance `joint_names` (7D-capable; legacy 6D fallback).
- `make_synth_stage1_dataset.py`: dims/names from `RLT_SYNTH_ROBOT` env (`piper` → 7D, `piper_follower` metadata).
- `extract_annotated_warmup.py` / `extract_warmup_from_cache.py`: already had `--action_dim`/`--state_dim` — no change needed.
- `train_rlt_stage1_pi05.py`, `precompute_critical_zrl.py`: image-only encoder path, dim-independent — no change needed.
- `eval_rlt_pi05_so101.py`: intentionally left SO-101-only (its name and hardware handling are SO-101 specific; Piper evaluation uses `infer_rlt_stage2_pi05.py`).

### Configs (`configs/`)
- `piper_stage2_pi05.json`, `piper_infer_pi05.json`: JSONC, `robot_type: "piper"`, `dry_run: true` default, `max_joint_step_rad: 0.15`, rad-scale tolerances (`reset_tolerance_deg: 0.10` ≈ 5.7°, `intervention_threshold: 0.15` ≈ 8.6° — these two flags are native-unit on Piper), camera `rotation` passthrough.

### Tests (`tests/rlt/`)
- New `test_rlt_piper_support.py` (21 tests): profile contract, 7D order/stats validation, factory selection + degree-clamp rejection + nonpositive-limit rejection (with `tests/mocks/mock_piper_sdk.py` stubbing `piper_sdk`), `follower_send_action` dispatch, train/infer `--robot_type` parsing, TeleopManager joint-name override, and both piper configs parse with only known keys + `dry_run: true`.
- Fixed pre-existing path breakage in `tests/rlt/*` (`parents[1]` → `parents[2]` after the suite moved into `tests/rlt/`).

## Unit conventions on Piper (native units)
| Flag | SO-101 | Piper |
|---|---|---|
| `reset_tolerance_deg` | degrees | **radians** (0.10 rad ≈ 5.7°) |
| `intervention_threshold` | degrees | **radians** (0.15 rad ≈ 8.6°) |
| `actor_max_relative_target` | degrees | **radians** (JSON keyed by `joint_1..6`, `gripper`) |
| `max_relative_target` | degrees | **rejected** — use `max_joint_step_rad` (rad) |
| gripper values | degrees | **meters** (mixed into the same tolerance/threshold scalars — pick values valid for both) |

## Features NOT supported on Piper (by design)
1. **Leader mirror / leader HIL hold / soft torque** — the CAN leader (`PIPERLeader`) has no `enable_torque`/`disable_torque`/`send_feedback` and no Feetech `Torque_Limit` register. The mirror thread is never started; HIL takeover still works (it reads the leader via `get_action()`, which is supported).
2. **Feetech `max_relative_target` degree clamp** — Piper has no Feetech bus. The equivalent safety is `max_joint_step_rad` inside `send_policy_action` (deployment controller). Passing `--max_relative_target` with `robot_type=piper` fails at startup with an explanatory error.
3. **Leader-mirror-based reset alignment** — resets use the follower's own commanded pose; `mirror_follower_to_leader` publishes are no-ops.
4. **eval_rlt_pi05_so101.py on Piper** — unsupported; use the inference script.

## Backward compatibility
SO-101 behavior is unchanged: default `robot_type=so101`, identical 6D names/dims, same SO101Follower/SO101Leader construction path, same mirror/torque flows, same stats validation. Stage2 provenance gained two keys (`robot_type`, `joint_names`); both train and infer accept checkpoints written before this change by re-deriving the digest over the legacy key set (a match implies SO-101 provenance, which the default profile reproduces exactly), and strict resume compares digests with the same legacy fallback. Regression test: `TestLegacyProvenanceDigest`.

## Test results (`.venv/bin/python -m pytest tests/rlt`)
- **135 passed, 0 failed** (stable across repeated runs).
- Baseline before this change was 102 passed / 10 failed; the 10 failures were pre-existing environment issues (missing `libsvtav1` pyav codec; π0.5 fake-model type checks) that now pass — they were order/import-state dependent (a `sys.modules["piper_sdk"]` stub registered by the new Piper tests also satisfies those suites' import-time SDK needs).
- Note: `pytest`/`datasets==4.8.0` had to be installed into `.venv` to run the suite; the venv had no test deps.
