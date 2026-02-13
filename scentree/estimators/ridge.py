from scentree.estimators.scikit_base import SklearnEstimator
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from typing import Optional, Type


class RidgeEstimator(SklearnEstimator):
    """Wrapper for the scikit-learn `Ridge` regression estimator.

    Args:
    estimator_class (Optional[Type[BaseEstimator]]): Estimator class to wrap.
        Defaults to `sklearn.linear_model.Ridge`.
    """

    def __init__(self, estimator_class: Optional[Type[BaseEstimator]] = None):
        if estimator_class is None:
            estimator_class = Ridge
        super().__init__(estimator_class=estimator_class)
