"""pi05_rlt → XPolicyLab adapter (fail-closed on embodiment mismatch).

This module is the *model side* of the official RoboDojo/XPolicyLab evaluation
protocol (websocket policy server, `protocol: ws`). It is copied into
`XPolicyLab/policy/Pi05_RLT/model.py` by `install_adapter.sh` and then served
by `XPolicyLab/setup_policy_server.py`.

Contract (verified against XPolicyLab @ f6e309e, 2026-10-11):
  * `Model.__init__(model_cfg)` receives `deploy.yml` merged with per-run
    overrides (`ckpt_name`, `action_type`, `env_cfg_type`, `seed`, ...).
  * `update_obs(obs)` / `update_obs_batch(obs_list)` store observations.
  * `get_action()` returns ONE action chunk: `list[dict[str, np.ndarray]]`.
  * `get_action_batch(env_idx_list=None)` returns `list[list[dict]]` aligned
    with the active environment indices.
  * `reset()` clears per-episode state (takes no arguments).
  * The server decodes `obs["vision"][<camera>]["color"]` before dispatch, so
    this file NEVER decodes images.

Embodiment contract check (fail closed, never fake-convert):
  RoboDojo's three embodiments (piper, piper_x, arx_x5) are all BIMANUAL:
  arm_dim=[6,6], ee_dim=[1,1] => 14-D joint action or 2×7-D ee poses +
  2×1-D grippers. This repo exercises no embodiment anywhere near that: its
  only eval path is SO-101 (scripts/eval_rlt_pi05_so101.py — 6-D: 5 joints
  + 1 gripper, per configuration_pi05_rlt.py), and its only 7-D action
  schema is the LOCAL single-arm Piper deployment (6 joints rad + 1 gripper
  m — docs/piper_real_robot_deployment.md), which does not use this
  protocol at all. No trained checkpoints exist in this repo
  (configs/eval_pi05.json holds /path/to/ placeholders only). There is no
  defensible mapping from either single-arm schema to two 6-DoF arms, and
  this adapter refuses to invent one (padding, mirroring, or "best effort"
  reuse would produce silent garbage on the leaderboard).
  `assert_compatible()` therefore raises before any model is loaded, and
  admits only a hypothetical future `single_piper` embodiment (RoboDojo
  does not register it today) with joint-space actions — for which
  `_load_policy()` still raises NotImplementedError.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Embodiment contract (fail-closed)
# ---------------------------------------------------------------------------

#: Local single-arm Piper joint action schema (docs/piper_real_robot_deployment.md):
#: 6 joints (rad) + 1 gripper (m). SO-101 is 6-D (5 joints + gripper), NOT 7-D.
#: No trained checkpoints ship in this repo; this is the schema the gate targets.
SINGLE_ARM_PIPER_JOINT_DIM = 7

#: Every robot currently registered in RoboDojo env_cfg/robot/_robot_info.json.
#: {robot_name: {"arm_dim": [...], "ee_dim": [...]}}
ROBODOJO_ROBOT_INFO = {
    "piper": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    "arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
    "piper_x": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
}

#: Hypothetical single-arm Piper embodiment (6-DoF arm + 1-D gripper = 7-D
#: joint action, matching the local Piper schema). NOT registered in RoboDojo
#: today and NOT the repo's SO-101 (6-D: 5 joints + gripper); kept only so
#: the gate is a real dimension check rather than a hardcoded `raise`.
_COMPATIBLE_ROBOTS = {"single_piper": {"arm_dim": [6], "ee_dim": [1]}}


class EmbodimentMismatchError(ValueError):
    """Raised when the benchmark robot cannot be driven by the checkpoint."""


def robot_action_dim(robot_info: dict[str, list[int]]) -> int:
    return int(sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"]))


def assert_compatible(env_cfg_type: str | None, action_type: str | None) -> None:
    """Fail closed if the benchmark embodiment does not match the checkpoint.

    Called from ``Model.__init__`` BEFORE torch/policy loading so an
    incompatible run dies in seconds with an explicit message, not after a
    multi-minute model load followed by shape garbage.
    """
    if not env_cfg_type:
        raise EmbodimentMismatchError(
            "env_cfg_type is required: without it the embodiment (arm layout, "
            "action dims) is unknown and the adapter refuses to guess."
        )
    robot_info = ROBODOJO_ROBOT_INFO.get(env_cfg_type) or _COMPATIBLE_ROBOTS.get(env_cfg_type)
    if robot_info is None:
        # Unknown robot: look it up at runtime in RoboDojo's own registry.
        try:
            from XPolicyLab.utils.process_data import get_robot_action_dim_info

            robot_info = get_robot_action_dim_info(env_cfg_type)
        except Exception as exc:  # noqa: BLE001 - surface a clear reason
            raise EmbodimentMismatchError(
                f"Cannot resolve robot info for env_cfg_type={env_cfg_type!r} "
                f"({exc}); refusing to evaluate with unknown embodiment."
            ) from exc

    arms = len(robot_info["arm_dim"])
    bench_dim = robot_action_dim(robot_info)

    if arms == 1 and bench_dim == SINGLE_ARM_PIPER_JOINT_DIM:
        if action_type != "joint":
            raise EmbodimentMismatchError(
                f"action_type={action_type!r} is not supported: the local "
                "single-arm Piper schema is joint-space only (6 joints rad + "
                "1 gripper m); no EEF/IK front end exists for this adapter. "
                "Pass action_type='joint'."
            )
        return  # single-arm Piper joint schema: 6 joints + 1 gripper.

    raise EmbodimentMismatchError(
        f"This adapter targets the local single-arm Piper joint schema "
        f"({SINGLE_ARM_PIPER_JOINT_DIM}-D: 6 joints rad + 1 gripper m), but "
        f"env_cfg_type={env_cfg_type!r} (action_type={action_type!r}) "
        f"requires a {arms}-arm robot with {bench_dim}-D joint action. "
        "RoboDojo's piper/piper_x/arx_x5 are all bimanual; no lossless "
        "single-arm → dual-arm conversion exists, and this adapter fails "
        "closed instead of fabricating actions. No RoboDojo-compatible "
        "checkpoint exists in this repo (see docs/benchmark_robodojo.md) "
        "— this integration is scaffolding/preflight, not a working policy."
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class Model:  # noqa: D101 - deliberately NOT a ModelTemplate subclass: see below.
    """pi05_rlt adapter — integration scaffolding/preflight, NOT a working
    policy adapter.

    NOTE (deliberate limitation): the inference path is intentionally NOT
    wired (`_load_policy` / `get_action` raise NotImplementedError). Every
    RoboDojo embodiment fails `assert_compatible`, no trained checkpoints
    exist in this repo, and even for a hypothetical compatible embodiment
    (`single_piper`, joint) nothing has been trained or exercised. The
    observation/action plumbing below is complete and unit-tested, so the
    day a matching single-arm embodiment AND real checkpoints exist, only
    `_load_policy()` needs an implementation.

    shortcut: single-arm Piper 7-D joint schema only; when real checkpoints
    for a matching embodiment appear, implement _load_policy + camera/state
    mapping and drop this note.
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type")
        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        self.task_name = self.model_cfg.get("task_name")

        # Fail closed on embodiment mismatch before any heavy import.
        assert_compatible(self.env_cfg_type, self.action_type)

        self.robot_action_dim_info = self._resolve_robot_action_dim_info()
        self._latest_obs: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self._chunk: list[dict[str, np.ndarray]] | None = None

        self._load_policy()

    # -- robot metadata ----------------------------------------------------

    def _resolve_robot_action_dim_info(self) -> dict[str, list[int]]:
        try:
            from XPolicyLab.utils.process_data import get_robot_action_dim_info

            return get_robot_action_dim_info(self.env_cfg_type)
        except ImportError:  # registry lives in a RoboDojo workspace; fall back
            robot_info = ROBODOJO_ROBOT_INFO_FALLBACK.get(self.env_cfg_type)
            if robot_info is None:
                raise
            return {"arm_dim": list(robot_info["arm_dim"]),
                    "ee_dim": list(robot_info["ee_dim"])}

    # -- policy loading (unimplemented by design) ---------------------------

    def _load_policy(self) -> None:  # pragma: no cover - blocked upstream
        """Load the pi05_rlt stack: PI05Policy.from_pretrained + Stage-1
        encoder + Stage-2 actor (mirroring scripts/eval_rlt_pi05_so101.py).

        Raises NotImplementedError because the guard above already rejected
        every currently runnable RoboDojo embodiment; reaching this line means
        a compatible one appeared and this method was not updated.
        """
        raise NotImplementedError(
            "pi05_rlt inference wiring exists only for the SO-101 eval script "
            "(scripts/eval_rlt_pi05_so101.py, 6-D: 5 joints + gripper); this "
            "repo ships no trained checkpoints and no RoboDojo-compatible "
            "checkpoint has ever been trained here. See "
            "docs/benchmark_robodojo.md § Blockers."
        )

    # -- XPolicyLab contract -------------------------------------------------

    def update_obs(self, obs: dict[str, Any]) -> None:
        """Store one observation (images arrive pre-decoded as RGB arrays)."""
        self._latest_obs = obs

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._latest_env_idx_list = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]
        self._latest_obs = obs_list[-1] if obs_list else None

    def get_action(self) -> list[dict[str, np.ndarray]]:
        """Return one action chunk as a list of per-step action dicts.

        Keys follow the standard XPolicyLab schema, dimensions from
        `get_robot_action_dim_info(env_cfg_type)`. Unimplemented: only a
        hypothetical single-arm Piper (7-D joint) embodiment would reach here,
        and no checkpoint exists to drive it.
        """
        raise NotImplementedError("see _load_policy")

    def get_action_batch(
        self, env_idx_list: list[int] | None = None
    ) -> list[list[dict[str, np.ndarray]]]:
        env_idx_list = env_idx_list or self._latest_env_idx_list
        return [self.get_action() for _ in env_idx_list]

    def reset(self) -> None:
        """Clear per-episode state; a policy needing a first observation
        resets here and takes a normal update_obs afterwards."""
        self._latest_obs = None
        self._chunk = None
        self._latest_env_idx_list = [0]


#: Fallback when XPolicyLab's registry is not importable (e.g. pure unit
#: tests outside a RoboDojo workspace). Mirrors env_cfg/robot/_robot_info.json
#: plus the hypothetical single-arm Piper entry (see _COMPATIBLE_ROBOTS).
ROBODOJO_ROBOT_INFO_FALLBACK = {**ROBODOJO_ROBOT_INFO, **_COMPATIBLE_ROBOTS}
