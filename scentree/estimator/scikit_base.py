import numpy as np
from scentree.estimator.utils import get_default_parameters, get_hyperparameters_space
from scentree.metrics.rmse import rmse
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
from typing import Any, Dict, Type, TypeVar, Union

R = TypeVar("R", bound="SklearnEstimator")


class SklearnEstimator(BaseEstimator):
    """Wrapper class for Scikit-learn estimators.

    This class provides a unified interface to integrate Scikit-learn
    estimators into the scentree framework. It allows dynamic creation
    of estimators with specified hyperparameters and exposes
    Scikit-learn-compatible methods such as `fit`, `predict`,
    `get_params`, and `set_params`.

    Attributes:
        estimator_class (Type[Any]): The Scikit-learn
            estimator class to be wrapped (e.g., `sklearn.linear_model.Ridge`).
        estimator (Optional[Any]): Instance of the fitted
            estimator. Set after calling `fit`.
        name (str): Name of the estimator class (derived from `estimator_class.__name__`).
        hyperparameters (Dict[str, Any]): Dictionary of hyperparameter values.
        hyperparameters_space (Dict[str, List[Any]]): Dictionary defining the search space
            for hyperparameters.
    """

    def __init__(self, estimator_class: Type[Any]):
        self.estimator = None
        self.estimator_class = estimator_class
        self.name = self.estimator_class.__name__
        self.hyperparameters = get_default_parameters(estimator_class)
        self.hyperparameters_space = get_hyperparameters_space(self.name)

    def fit(self: R, X: np.ndarray, y: np.ndarray) -> R:
        """Fit the wrapped Scikit-learn estimator to the training data.

        Args:
            X (np.ndarray): Input feature matrix for training.
            y (np.ndarray): Target vector.

        Returns:
            R: The fitted wrapper instance.
        """

        self.estimator = self.estimator_class(**self.hyperparameters)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted Scikit-learn estimator.

        Args:
            X (np.ndarray): Input feature matrix for prediction.

        Raises:
            RuntimeError: If `fit()` has not been called before prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.estimator is None:
            raise RuntimeError("You must call `fit()` before `predict()`.")
        return self.estimator.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the hyperparameters of the estimator.

        Args:
            deep (bool): Ignored parameter for compatibility with Scikit-learn.
                Included for consistency with the BaseEstimator API.

        Returns:
            Dict[str, Any]: Dictionary of current hyperparameter values.
        """
        return {
            "estimator_class": self.estimator_class,
            "hyperparameters": self.hyperparameters,
            "hyperparameters_space": self.hyperparameters_space,
        }

    def set_params(self: R, **params) -> R:
        """Set hyperparameter values for the estimator.

        Args:
            **params: Arbitrary keyword arguments of hyperparameters to update.

        Returns:
            R: The updated estimator instance (same type as self).
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.hyperparameters:
                self.hyperparameters[key] = value
            else:
                raise ValueError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}. "
                    f"Valid parameters: constructor={list(self.__dict__.keys())}, "
                    f"hyperparameters={list(self.hyperparameters.keys())}."
                )
        return self

    def get_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Union[int, _BaseKFold] = 5,
    ) -> BaseEstimator:
        """
        Perform a Grid Search over the estimator's hyperparameters
        and return a trained instance of the best model.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training target vector or matrix.
            cv (int or BaseCrossValidator): Cross-validation strategy. Default is 5-fold.

        Returns:
            BaseEstimator: An instance of the estimator with the best
            hyperparameters, already fitted to the training data.
        """
        # Define scoring, compatible with multi-output
        scorer = make_scorer(rmse, greater_is_better=False)

        if isinstance(cv, int):
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = cv

        # Create GridSearchCV
        grid = GridSearchCV(
            estimator=self.estimator_class(),
            param_grid=self.hyperparameters_space,
            scoring=scorer,
            cv=cv_splitter,
            n_jobs=-1,
        )

        # Fit the grid search
        grid.fit(X, y)

        # Instantiate and fit the best estimator
        best_estimator = self.estimator_class(**grid.best_params_)
        best_estimator.fit(X, y)

        return best_estimator
