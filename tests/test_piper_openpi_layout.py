import pytest

from piper_openpi.layout import piper_layout
from piper_openpi.piper_contract import delta_action_mask


@pytest.mark.parametrize(
    ("layout", "dimension", "grippers"),
    [("left_arm", 7, (6,)), ("right_arm", 7, (6,)), ("dual_arm", 14, (6, 13))],
)
def test_existing_layout_contract(layout, dimension, grippers):
    spec = piper_layout(layout)
    mask = delta_action_mask(layout)
    assert spec["state_dim"] == spec["action_dim"] == dimension
    assert len(mask) == dimension
    assert [i for i, is_delta in enumerate(mask) if not is_delta] == list(grippers)


def test_schema_aliases_use_the_same_existing_layouts():
    assert piper_layout("left_arm_7") == piper_layout("left_arm")
    assert piper_layout("dual_arm_14")["action_dim"] == 14
    with pytest.raises(ValueError, match="Unsupported Piper layout"):
        piper_layout("single")
