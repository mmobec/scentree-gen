import numpy as np
import pytest
from scentree.fan_generator.stage_manager import StageManager


def test_clip_matrix_values_without_ranges_returns_same_matrix() -> None:
    manager = StageManager()

    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = manager.clip_matrix_values(X, None)

    np.testing.assert_array_equal(result, X)


def test_generate_scenario_fans_invalid_value_ranges() -> None:
    manager = StageManager()

    X = np.random.rand(10, 2)

    with pytest.raises(ValueError):
        manager.generate_scenario_fans(
            X=X,
            num_fans=2,
            num_scenarios=3,
            value_ranges=[(0.0, 1.0)],
        )
