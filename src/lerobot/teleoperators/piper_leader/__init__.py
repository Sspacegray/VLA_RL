from typing import TYPE_CHECKING, Any

from .config_piper_leader import PIPERLeaderConfig

if TYPE_CHECKING:
    from .piper_leader import PIPERLeader


def __getattr__(name: str) -> Any:
    """Keep the config importable (subclass registration) without the hardware SDK."""

    if name == "PIPERLeader":
        from .piper_leader import PIPERLeader

        return PIPERLeader
    raise AttributeError(name)
