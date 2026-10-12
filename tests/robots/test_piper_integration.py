"""Piper registration and inference/record integration on the lerobot 0.6.2 base.

Covers: factories dispatch, script imports, the policy send path
(``send_policy_action`` preferred in rollout + async client), the
record-with-follower-feedback scheme, the ``piper`` extra, and configurable
CAN names. All hardware-free — ``tests/mocks/mock_piper_sdk.py`` stubs the SDK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "tests" / "mocks"))
import mock_piper_sdk  # noqa: E402

sys.modules.setdefault("piper_sdk", mock_piper_sdk)


def _grpcio_missing() -> bool:
    try:
        import grpc  # noqa: F401
    except ImportError:
        return True
    return False


from lerobot.robots.piper_follower.config_piper_follower import PIPERFollowerConfig  # noqa: E402
from lerobot.teleoperators.piper_leader.config_piper_leader import PIPERLeaderConfig  # noqa: E402


def _feedback(values: list[float]) -> dict[str, float]:
    return {
        **{f"joint_{i + 1}.pos": float(values[i]) for i in range(6)},
        "gripper.pos": float(values[6]),
    }


class _Recording:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def send_action(self, action):
        self.calls.append(("send_action", action))
        return action

    def send_policy_action(self, action):
        self.calls.append(("send_policy_action", action))
        return action


# ── Factory registration ───────────────────────────────────────────────────


def test_make_robot_from_config_dispatches_piper_follower(monkeypatch):
    import lerobot.robots.piper_follower as follower_pkg
    from lerobot.robots.utils import make_robot_from_config

    created = {}

    class Fake:
        def __init__(self, config):
            created["config"] = config

    monkeypatch.setattr(follower_pkg, "PIPERFollower", Fake)
    config = follower_pkg.PIPERFollowerConfig(cameras={})
    assert config.type == "piper_follower"
    robot = make_robot_from_config(config)
    assert robot.__class__.__name__ == "Fake"
    assert isinstance(created["config"], follower_pkg.PIPERFollowerConfig)


def test_make_teleoperator_from_config_dispatches_piper_leader(monkeypatch):
    import lerobot.teleoperators.piper_leader as leader_pkg
    from lerobot.teleoperators.utils import make_teleoperator_from_config

    created = {}

    class Fake:
        def __init__(self, config):
            created["config"] = config

    monkeypatch.setattr(leader_pkg, "PIPERLeader", Fake)
    config = leader_pkg.PIPERLeaderConfig()
    assert config.type == "piper_leader"
    make_teleoperator_from_config(config)
    assert isinstance(created["config"], leader_pkg.PIPERLeaderConfig)


def test_piper_configs_register_in_choice_registry():
    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    assert RobotConfig.get_choice_class("piper_follower") is PIPERFollowerConfig
    assert TeleoperatorConfig.get_choice_class("piper_leader") is PIPERLeaderConfig


# ── Script imports ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "module_name",
    [
        "lerobot.scripts.lerobot_record",
        "lerobot.scripts.lerobot_teleoperate",
        "lerobot.scripts.lerobot_calibrate",
        "lerobot.scripts.lerobot_replay",
        pytest.param("lerobot.scripts.lerobot_rollout", marks=pytest.mark.skipif(
            _grpcio_missing(), reason="lerobot_rollout needs grpcio (async extra)"
        )),
    ],
)
def test_entry_scripts_import_piper(module_name):
    module = pytest.importorskip(module_name)
    assert hasattr(module, "piper_follower")
    if module_name != "lerobot.scripts.lerobot_replay":  # replay drives the robot only
        assert hasattr(module, "piper_leader")


# ── Inference path prefers send_policy_action ───────────────────────────────


def test_robot_wrapper_prefers_send_policy_action():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    class FakeRobot:
        name = "piper_follower"

        def send_action(self, action):
            raise AssertionError("policy actions must not use send_action")

        def send_policy_action(self, action):
            return action

    wrapper = ThreadSafeRobot(FakeRobot())
    action = _feedback([0.1] * 6 + [0.02])
    assert wrapper.send_policy_action(action) == action


def test_robot_wrapper_falls_back_to_send_action():
    from lerobot.rollout.robot_wrapper import ThreadSafeRobot

    class FakeRobot:
        name = "so101_follower"

        def send_action(self, action):
            return action

    wrapper = ThreadSafeRobot(FakeRobot())
    action = _feedback([0.1] * 6 + [0.02])
    assert wrapper.send_policy_action(action) == action


def test_rollout_send_next_action_uses_policy_path():
    import torch

    from lerobot.rollout.strategies.core import send_next_action

    recording = _Recording()
    action = _feedback([0.1] * 6 + [0.02])

    class FakeInterpolator:
        def add(self, tensor):
            pass

        def get(self):
            return torch.tensor(list(action.values()))

        def needs_new_action(self):
            return False

    class FakeProcessors:
        class _P:
            def __call__(self, payload):
                return payload[0]

        teleop_action_processor = robot_action_processor = robot_observation_processor = _P()

    class FakeHardware:
        robot_wrapper = recording

    class FakeCtx:
        processors = FakeProcessors()
        hardware = FakeHardware()

        class data:
            dataset_features = {}
            ordered_action_keys = list(action.keys())

        class policy:
            class inference:
                @staticmethod
                def get_action(obs_frame):
                    return None

    class FakeSection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeTimer:
        def section(self, name):
            return FakeSection()

    result = send_next_action(
        obs_processed={}, obs_raw={}, ctx=FakeCtx(), interpolator=FakeInterpolator(), timer=FakeTimer()
    )
    assert result is not None
    assert [name for name, _ in recording.calls] == ["send_policy_action"]
    for key, value in action.items():
        assert result[key] == pytest.approx(value, rel=1e-6)


def test_async_client_control_loop_prefers_policy_path():
    import inspect

    pytest.importorskip("grpc")  # async client module needs grpcio
    from lerobot.async_inference import robot_client as rc

    src = inspect.getsource(rc.RobotClient.control_loop_action)
    assert "send_policy_action" in src
    assert "send_action" in src


# ── Record path preserves follower feedback ─────────────────────────────────


def test_record_loop_records_follower_feedback(monkeypatch):
    import lerobot.scripts.lerobot_record as record_mod
    from lerobot.teleoperators.teleoperator import Teleoperator
    from lerobot.utils.feature_utils import hw_to_dataset_features

    commanded = _feedback([0.5] * 6 + [0.05])
    measured = _feedback([0.1] * 6 + [0.02])

    class FakeConfig:
        record_action_from_follower = True

    class FakeRobot:
        name = "piper_follower"
        config = FakeConfig()
        def get_observation(self):
            return dict(measured)

        def send_action(self, action):
            return action

        def get_record_action_from_follower(self):
            return dict(measured)

    class FakeTeleop(Teleoperator):
        def __init__(self):
            pass

        def get_action(self):
            return dict(commanded)

        # remaining abstract members are unused by record_loop
        def connect(self, *a, **k):
            pass

        def calibrate(self, *a, **k):
            pass

        def configure(self, **k):
            pass

        def disconnect(self):
            pass

        def send_feedback(self, *a, **k):
            pass

        @property
        def is_connected(self):
            return True

        @property
        def is_calibrated(self):
            return True

        @property
        def action_features(self):
            return {k: float for k in commanded}

        @property
        def feedback_features(self):
            return {}

    robot = FakeRobot()
    teleop = FakeTeleop()

    class FakeTimer:
        def tick(self):
            pass

        def section(self, name):
            return _NullSection()

        def wait(self):
            raise _Stop()

        def note_starved_tick(self):
            pass

    class _NullSection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _Stop(Exception):
        pass

    frames = []

    class FakeDataset:
        fps = 30
        features = hw_to_dataset_features(
            {f"joint_{i}.pos": float for i in range(1, 7)} | {"gripper.pos": float}, prefix="action"
        )

        def add_frame(self, frame):
            frames.append(frame)

    class _Proc:
        def __call__(self, payload):
            return payload[0] if isinstance(payload, tuple) else payload

    class _Procs:
        teleop_action_processor = robot_action_processor = robot_observation_processor = _Proc()

    with pytest.raises(_Stop):
        record_mod.record_loop(
            robot=robot,
            events={"exit_early": False},
            fps=30,
            teleop_action_processor=_Procs.teleop_action_processor,
            robot_action_processor=_Procs.robot_action_processor,
            robot_observation_processor=_Procs.robot_observation_processor,
            dataset=FakeDataset(),
            teleop=teleop,
            control_time_s=10.0,
            single_task="test",
            display_data=False,
            display_mode=None,
            display_compressed_images=False,
            timer=FakeTimer(),
        )
    assert frames
    np.testing.assert_allclose(frames[0]["action"], [0.1] * 6 + [0.02], rtol=1e-6)


# ── pyproject piper extra ───────────────────────────────────────────────────


def test_pyproject_declares_piper_extra():
    from lerobot.robots.piper_follower.config_piper_follower import PIPERFollowerConfig

    text = (REPO / "pyproject.toml").read_text()
    assert 'piper = ["piper-sdk' in text
    assert "lerobot[piper]" in text


# ── Configurable CAN names ──────────────────────────────────────────────────


def test_can_names_are_configurable_with_defaults():
    from lerobot.robots.piper_follower.config_piper_follower import PIPERFollowerConfig
    from lerobot.teleoperators.piper_leader.config_piper_leader import PIPERLeaderConfig

    assert PIPERFollowerConfig(cameras={}).can_name == "can_follower"
    assert PIPERLeaderConfig().can_name == "can_master"
    assert PIPERFollowerConfig(cameras={}, can_name="can1").can_name == "can1"
