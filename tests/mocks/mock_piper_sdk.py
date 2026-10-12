"""Minimal piper_sdk stub so Piper classes import without the CAN SDK.

Registered into ``sys.modules["piper_sdk"]`` by tests/rlt/test_rlt_piper_support.py;
never imported directly. Only the surface touched by
``lerobot.motors.piper.piper`` at import time is provided.
"""

class C_PiperInterface_V2:  # noqa: N801 (match the real SDK name)
    """Import-time placeholder; instances are never created in tests."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("mock piper_sdk cannot talk to hardware")
