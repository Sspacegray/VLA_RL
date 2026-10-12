"""Look up the Piper arm layout already defined by the OpenPI profile.

The aliases match ``examples/robots/piper/local_runtime.py``; this does not
define a new robot or crop one checkpoint into another.
"""

from __future__ import annotations

from piper_openpi.profile import PROFILE

_ALIASES = {
    "left_arm_7": "left_arm",
    "right_arm_7": "right_arm",
    "dual_arm_14": "dual_arm",
}


def piper_layout(layout_id: str) -> dict:
    key = _ALIASES.get(layout_id, layout_id)
    for item in PROFILE["layouts"]:
        if item["id"] == key:
            return item
    known = sorted(_ALIASES) + [item["id"] for item in PROFILE["layouts"]]
    raise ValueError(f"Unsupported Piper layout: {layout_id}; expected one of {known}")
