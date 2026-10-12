"""Dependency-free action contract shared by Piper training and data tooling."""

from __future__ import annotations


def delta_action_mask(layout: str) -> tuple[bool, ...]:
    """Use joint deltas and absolute normalized gripper targets."""

    masks = {
        "left_arm": (True,) * 6 + (False,),
        "right_arm": (True,) * 6 + (False,),
        "dual_arm": (True,) * 6 + (False,) + (True,) * 6 + (False,),
    }
    try:
        return masks[layout]
    except KeyError as exc:
        raise ValueError(f"Unsupported Piper layout: {layout}") from exc


__all__ = ["delta_action_mask"]
