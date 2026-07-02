import numpy as np
import pytest
from scentree.tree_construction.ftc import FTC


def test_validate_scenarios_raises_when_rows_do_not_match() -> None:
    scenario1 = np.zeros((3, 2))
    scenario2 = np.zeros((4, 2))

    with pytest.raises(ValueError):
        FTC(
            scenarios=[scenario1, scenario2],
            stage_ids=[1],
            num_variables_per_stage=[2],
        )


def test_validate_consistency_raises_when_lengths_do_not_match() -> None:
    scenario = np.zeros((3, 2))

    with pytest.raises(ValueError):
        FTC(
            scenarios=[scenario],
            stage_ids=[1, 2],
            num_variables_per_stage=[2],
        )
