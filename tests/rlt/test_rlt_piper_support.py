"""Piper 7D support in the shared RLT core and entry scripts.

Pins the robot-profile contract (names/dims/capabilities), the parameterized
order/stats helpers, the lazy hardware factory (mocked — no piper_sdk needed),
and that the shipped configs/piper_*.json parse with only known keys.
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

# Stub the CAN SDK so the Piper classes import on machines without piper_sdk.
sys.path.insert(0, str(Path(__file__).parents[2] / "tests" / "mocks"))
import mock_piper_sdk  # noqa: E402

sys.modules.setdefault("piper_sdk", mock_piper_sdk)

from rlt_core import (  # noqa: E402
    PIPER_JOINT_NAMES,
    ROBOT_PROFILES,
    SO101_JOINT_NAMES,
    _validate_quantiles,
    action_order_indices,
    follower_send_action,
    load_normalization_stats,
    make_robot_pair,
    reorder_action_tensor,
    robot_profile,
)


# ── Robot profiles ──────────────────────────────────────────────────────────

def test_piper_profile_is_seven_dim_and_capability_gated():
    profile = robot_profile("piper")
    assert profile.joint_names == PIPER_JOINT_NAMES
    assert PIPER_JOINT_NAMES == ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper")
    assert profile.action_dim == 7 and profile.state_dim == 7
    # Capability flags: the CAN leader has no Feetech torque/feedback path and
    # no serial port; policy commands go through send_policy_action.
    assert not profile.supports_leader_mirror
    assert not profile.supports_torque_control
    assert not profile.uses_serial_port
    assert profile.policy_action_method == "send_policy_action"


def test_so101_profile_is_the_backward_compatible_default():
    assert robot_profile(None).joint_names == SO101_JOINT_NAMES
    so101 = robot_profile("so101")
    assert so101.action_dim == 6
    assert so101.supports_leader_mirror and so101.supports_torque_control
    assert so101.policy_action_method == "send_action"


def test_unknown_robot_type_fails_closed():
    with pytest.raises(ValueError, match="Unknown robot_type"):
        robot_profile("aloha")


# ── Action ordering with an explicit allowed set ────────────────────────────

def test_action_order_indices_accepts_piper_joint_set():
    shuffled = list(PIPER_JOINT_NAMES)[::-1]
    indices = action_order_indices(source_names=PIPER_JOINT_NAMES, target_names=shuffled, allowed=PIPER_JOINT_NAMES)
    positions = {name: i for i, name in enumerate(PIPER_JOINT_NAMES)}
    assert indices == tuple(positions[name] for name in shuffled)


def test_action_order_indices_rejects_piper_names_under_so101_default():
    with pytest.raises(ValueError, match="must contain"):
        action_order_indices(list(PIPER_JOINT_NAMES), list(PIPER_JOINT_NAMES))


def test_reorder_action_tensor_round_trips_piper_7d():
    action = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
    flipped = list(PIPER_JOINT_NAMES)[::-1]
    reordered = reorder_action_tensor(
        action, source_names=PIPER_JOINT_NAMES, target_names=flipped, allowed=PIPER_JOINT_NAMES,
    )
    assert reordered.tolist() == [[[6, 5, 4, 3, 2, 1, 0]]]
    # SO-101 default allowed set is unchanged.
    so101 = torch.arange(6, dtype=torch.float32)
    assert reorder_action_tensor(so101, source_names=SO101_JOINT_NAMES, target_names=SO101_JOINT_NAMES).tolist() == [0, 1, 2, 3, 4, 5]


# ── Quantile validation at 7D ───────────────────────────────────────────────

def test_validate_quantiles_accepts_7d_when_expected():
    stats = {
        "observation.state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
    }
    result = _validate_quantiles(stats, expected_dim=7)
    assert len(result["action"]["q01"]) == 7


def test_validate_quantiles_rejects_6d_when_7d_expected():
    stats = {
        "observation.state": {"q01": [0.0] * 6, "q99": [1.0] * 6},
        "action": {"q01": [0.0] * 6, "q99": [1.0] * 6},
    }
    with pytest.raises(ValueError, match="exactly 7D"):
        _validate_quantiles(stats, expected_dim=7)


def test_validate_quantiles_default_stays_6d():
    stats = {
        "observation.state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "action": {"q01": [0.0] * 7, "q99": [1.0] * 7},
    }
    with pytest.raises(ValueError, match="exactly 6D"):
        _validate_quantiles(stats)


def test_load_normalization_stats_validates_at_piper_dim(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "observation.state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "action": {"q01": [-1.0] * 7, "q99": [1.0] * 7},
        "use_relative_actions": False,
    }))
    stats, source = load_normalization_stats(tmp_path, str(stats_path), 7)
    assert source.endswith("stats.json")
    assert len(stats["action"]["q99"]) == 7


# ── Hardware factory (mocked; piper_sdk / feetech never imported) ───────────

class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.deployment = kwargs.get("deployment")


def test_make_robot_pair_builds_piper_with_deployment_limit():
    made = {}

    class FakeFollower:
        def __init__(self, config):
            made["follower"] = config

    class FakeLeader:
        def __init__(self, config):
            made["leader"] = config

    import lerobot.robots.piper_follower.piper_follower as follower_mod
    import lerobot.teleoperators.piper_leader.piper_leader as leader_mod
    from lerobot.robots.piper_follower.deployment import PiperDeploymentConfig

    with mock.patch.object(follower_mod, "PIPERFollower", FakeFollower), \
         mock.patch.object(leader_mod, "PIPERLeader", FakeLeader):
        follower, leader = make_robot_pair("piper", camera_configs={}, max_joint_step_rad=0.15)

    assert isinstance(follower, FakeFollower) and isinstance(leader, FakeLeader)
    deployment = made["follower"].deployment
    assert isinstance(deployment, PiperDeploymentConfig) and deployment.max_joint_step_rad == 0.15


def test_make_robot_pair_piper_keeps_deployment_default():
    """No --max_joint_step_rad: the config's default_factory must survive."""
    made = {}

    class FakeFollower:
        def __init__(self, config):
            made["config"] = config

    from lerobot.robots.piper_follower import piper_follower as follower_mod
    from lerobot.teleoperators.piper_leader import piper_leader as leader_mod

    class FakeLeader:
        def __init__(self, config):
            pass

    with mock.patch.object(follower_mod, "PIPERFollower", FakeFollower), \
         mock.patch.object(leader_mod, "PIPERLeader", FakeLeader):
        follower, _ = make_robot_pair("piper", camera_configs={})
    from lerobot.robots.piper_follower.deployment import PiperDeploymentConfig
    assert isinstance(follower, FakeFollower)
    assert isinstance(made["config"].deployment, PiperDeploymentConfig)
    assert made["config"].deployment.max_joint_step_rad is None


def test_make_robot_pair_rejects_degree_clamp_on_piper():
    with pytest.raises(ValueError, match="max_relative_target"):
        make_robot_pair("piper", camera_configs={}, max_relative_target=5.0)


def test_make_robot_pair_rejects_nonpositive_step_limit():
    with pytest.raises(ValueError, match="max_joint_step_rad"):
        make_robot_pair("piper", camera_configs={}, max_joint_step_rad=0.0)


def test_follower_send_action_prefers_policy_path():
    calls = []

    class PiperLike:
        def send_policy_action(self, action):
            calls.append(("policy", action))
            return action

        def send_action(self, action):
            calls.append(("teleop", action))
            return action

    class So101Like:
        def send_action(self, action):
            calls.append(("teleop", action))
            return action

    follower_send_action(PiperLike(), {"joint_1.pos": 0.0})
    follower_send_action(So101Like(), {"shoulder_pan.pos": 0.0})
    assert [kind for kind, _ in calls] == ["policy", "teleop"]


# ── Entry-script wiring ─────────────────────────────────────────────────────

def _load_script(name):
    root = Path(__file__).parents[2] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTrainScriptPiperWiring:
    def setup_method(self):
        self.module = _load_script("train_rlt_stage2_pi05.py")

    def _args(self, robot_type="piper", **extra):
        argv = [
            "train_rlt_stage2_pi05.py", "--pi05_path", "p", "--rlt_checkpoint", "c",
            "--robot_type", robot_type, "--dry_run",
        ]
        for key, value in extra.items():
            flag = key if key.startswith("-") else f"--{key}"
            if isinstance(value, bool):
                argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        with mock.patch.object(self.module.sys, "argv", argv):
            return self.module.parse_args()

    def test_robot_type_arg_parses_and_defaults_to_so101(self):
        args = self._args(robot_type="so101")
        assert args.robot_type == "so101"
        assert args.max_joint_step_rad is None

    def test_piper_args_parse(self):
        args = self._args(max_joint_step_rad="0.15")
        assert args.robot_type == "piper"
        assert args.max_joint_step_rad == 0.15

    def test_teleop_manager_joint_names_override(self):
        teleop = self.module.TeleopManager(leader=None, joint_names=PIPER_JOINT_NAMES)
        assert teleop.JOINT_NAMES == list(PIPER_JOINT_NAMES)
        # Default (no override) keeps SO-101 names.
        assert self.module.TeleopManager.JOINT_NAMES == list(SO101_JOINT_NAMES)


class TestInferScriptPiperWiring:
    def setup_method(self):
        self.module = _load_script("infer_rlt_stage2_pi05.py")

    def test_robot_type_arg_parses(self):
        argv = ["infer_rlt_stage2_pi05.py", "--pi05_path", "p", "--rlt_checkpoint", "c",
                "--resume_from", "s", "--robot_type", "piper", "--dry_run"]
        with mock.patch.object(self.module.sys, "argv", argv):
            args = self.module.parse_args()
        assert args.robot_type == "piper"

    def test_teleop_manager_joint_names_override(self):
        teleop = self.module.TeleopManager(leader=None, joint_names=PIPER_JOINT_NAMES)
        assert teleop.JOINT_NAMES == list(PIPER_JOINT_NAMES)


# ── Shipped configs ─────────────────────────────────────────────────────────

def _strip_json_comments(text):
    # The scripts' _strip_json_comments handles // and /* */ everywhere;
    # reuse the same rules (it strips before json.loads). Simple regex version
    # sufficient for these files: full-line //, block /* */, trailing //.
    import re
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|[^:\"\w])//.*$", "", text, flags=re.M)
    return text


@pytest.mark.parametrize("config_name,script_name", [
    ("piper_stage2_pi05.json", "train_rlt_stage2_pi05.py"),
    ("piper_infer_pi05.json", "infer_rlt_stage2_pi05.py"),
])
def test_piper_configs_use_only_known_keys(config_name, script_name):
    config_path = Path(__file__).parents[2] / "configs" / config_name
    values = json.loads(_strip_json_comments(config_path.read_text()))
    module = _load_script(script_name)
    parser = module.parse_args.__wrapped__ if hasattr(module.parse_args, "__wrapped__") else None
    # Rebuild the parser by introspecting: parse with --help suppressed is
    # intrusive; instead parse the config the same way the script does.
    argv = [script_name, "--config", str(config_path)]
    with mock.patch.object(module.sys, "argv", argv):
        args = module.parse_args()  # unknown keys -> parser.error -> SystemExit
    assert args.robot_type == "piper"
    assert values["robot_type"] == "piper"
    assert values["max_joint_step_rad"] == 0.15
    assert values.get("dry_run") is True
