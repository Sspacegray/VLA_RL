"""RoboDojo adapter contract tests (torch-free).

Pins the fail-closed embodiment gate and the XPolicyLab adapter plumbing
without importing torch or the VLA_RL stack (the adapter defers all heavy
imports to _load_policy, which the gate precedes).
"""
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "robodojo"
sys.path.insert(0, str(BENCH / "xpolicylab_adapter" / "pi05_rlt"))

import model as adapter  # noqa: E402
from model import (  # noqa: E402
    EmbodimentMismatchError,
    Model,
    assert_compatible,
)


# -- fail-closed gate -------------------------------------------------------


@pytest.mark.parametrize(
    "env_cfg", ["arx_x5", "piper", "piper_x"]
)
def test_dual_arm_robots_fail_closed(env_cfg):
    with pytest.raises(EmbodimentMismatchError, match="bimanual"):
        assert_compatible(env_cfg, "joint")


def test_dual_arm_ee_action_type_also_fails_closed():
    with pytest.raises(EmbodimentMismatchError):
        assert_compatible("piper", "ee")


def test_missing_env_cfg_fails_closed():
    with pytest.raises(EmbodimentMismatchError, match="env_cfg_type is required"):
        assert_compatible(None, "joint")
    with pytest.raises(EmbodimentMismatchError):
        assert_compatible("", "joint")


def test_unknown_robot_fails_closed_without_xpolicylab(monkeypatch):
    monkeypatch.setattr(
        sys, "path", [p for p in sys.path if "XPolicyLab" not in p]
    )
    with pytest.raises(EmbodimentMismatchError, match="Cannot resolve"):
        assert_compatible("franka", "joint")


def test_unknown_robot_fails_closed_when_registry_unavailable():
    import builtins

    real_import = builtins.__import__

    def no_xpolicylab(name, *args, **kwargs):
        if name.startswith("XPolicyLab"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=no_xpolicylab):
        with pytest.raises(EmbodimentMismatchError, match="Cannot resolve"):
            assert_compatible("franka", "joint")


def test_hypothetical_single_piper_joint_passes_gate():
    """Only the explicit local single-arm Piper 7-D joint schema passes.

    RoboDojo registers no such embodiment today; this pins the gate's
    dimension check so it is not a hardcoded `raise`.
    """
    assert_compatible("single_piper", "joint")


def test_so101_is_not_a_compatible_robot():
    """SO-101 is 6-D (5 joints + gripper), not a RoboDojo embodiment, and
    must not be smuggled through the gate as a 7-D single arm."""
    with pytest.raises(EmbodimentMismatchError, match="Cannot resolve"):
        assert_compatible("so101", "joint")


def test_compatible_robot_requires_joint_action_type():
    with pytest.raises(EmbodimentMismatchError, match="joint-space only"):
        assert_compatible("single_piper", "ee")


# -- Model plumbing ---------------------------------------------------------


def make_model(env_cfg="single_piper"):
    cfg = dict(env_cfg_type=env_cfg, action_type="joint", task_name="stack_bowls")
    with mock.patch.object(Model, "_load_policy"):
        return Model(cfg)


def test_model_init_passes_for_compatible_robot():
    m = make_model()
    assert m.env_cfg_type == "single_piper"
    assert m._latest_obs is None


def test_model_init_fails_closed_before_load():
    with pytest.raises(EmbodimentMismatchError):
        make_model("piper")


def test_model_load_policy_not_implemented_without_mock():
    """Even a compatible embodiment stops at the untrained inference path
    (scaffolding/preflight, not a working policy adapter)."""
    with pytest.raises(NotImplementedError, match="no trained checkpoints"):
        Model(dict(env_cfg_type="single_piper", action_type="joint"))


def test_model_robot_info_fallback():
    m = make_model()
    assert m.robot_action_dim_info["arm_dim"] == [6]
    assert m.robot_action_dim_info["ee_dim"] == [1]


def test_update_obs_stores_obs():
    m = make_model()
    obs = {"vision": {"0_left": {"color": np.zeros((3, 224, 224), np.uint8)}},
           "state": {}, "instruction": "pick"}
    m.update_obs(obs)
    assert m._latest_obs is obs


def test_update_obs_batch_tracks_env_idx():
    m = make_model()
    m.update_obs_batch([{"env_idx": 3, "state": {}}, {"env_idx": 1, "state": {}}])
    assert m._latest_env_idx_list == [3, 1]
    assert m._latest_obs["env_idx"] == 1


def test_get_action_batch_shape_and_reset():
    m = make_model()
    m.update_obs_batch([{"env_idx": 0, "state": {}}, {"env_idx": 2, "state": {}}])
    with pytest.raises(NotImplementedError):
        m.get_action_batch([0, 2])
    with pytest.raises(NotImplementedError):
        m.get_action_batch(None)  # falls back to tracked indices
    m.reset()
    assert m._latest_obs is None and m._chunk is None
    assert m._latest_env_idx_list == [0]


def test_deploy_contract_methods_exist():
    m = make_model()
    for name in ("update_obs", "update_obs_batch", "get_action",
                 "get_action_batch", "reset"):
        assert callable(getattr(m, name)), name


def test_deploy_files_static_checks():
    for f in ("model.py", "deploy.py", "__init__.py", "deploy.yml", "README.md"):
        assert (BENCH / "xpolicylab_adapter" / "pi05_rlt" / f).is_file(), f


def test_deploy_yml_standard_keys():
    import yaml
    cfg = yaml.safe_load((BENCH / "xpolicylab_adapter" / "pi05_rlt" / "deploy.yml").read_text())
    assert cfg["policy_name"] == "Pi05_RLT"
    assert cfg["protocol"] == "ws"
    for key in ("host", "port", "bench_name", "task_name", "ckpt_name",
                "env_cfg_type", "seed", "action_type", "gpu_id", "eval_batch"):
        assert key in cfg, key


def test_install_script_uses_real_checkpoints(tmp_path):
    """install_adapter.sh refuses missing checkpoint paths (no invented paths)."""
    import subprocess
    script = BENCH / "install_adapter.sh"
    robodojo = tmp_path / "RoboDojo"
    (robodojo / "XPolicyLab" / "policy").mkdir(parents=True)
    res = subprocess.run(
        ["bash", str(script), str(robodojo), "--pi05-path", str(tmp_path / "nope")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0
    assert "do not exist" in res.stderr


def test_install_script_requires_xpolicylab_policy_dir(tmp_path):
    import subprocess
    script = BENCH / "install_adapter.sh"
    res = subprocess.run(["bash", str(script), str(tmp_path)],
                         capture_output=True, text=True)
    assert res.returncode != 0
    assert "XPolicyLab" in res.stderr


def test_adapter_never_decodes_images():
    """CONTRIBUTING.md static check: model.py must not decode images."""
    src = (BENCH / "xpolicylab_adapter" / "pi05_rlt" / "model.py").read_text()
    for needle in ("cv2.imdecode", "np.frombuffer", "Image.open", "cv2.imencode"):
        assert needle not in src, needle
