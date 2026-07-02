import numpy as np
import pytest
from scentree.estimators.estimator_orchestrator import EstimatorController


def test_predict_without_fit_raises() -> None:
    controller = EstimatorController()

    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError):
        controller.predict(X)


def test_estimate_residuals_without_fit_raises() -> None:
    controller = EstimatorController()

    X = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError):
        controller.estimate_residuals(X)
