import numpy as np
from scentree.estimator.utils import get_default_parameters, get_hyperparameters_space
from scentree.metrics.rmse import rmse
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
from typing import Any, Dict, Optional, Type, TypeVar, Union

R = TypeVar("R", bound="SklearnEstimator")


class SklearnEstimator(BaseEstimator):
    """
    Wrapper class for scikit-learn estimators to integrate them into the Scentree framework.

    This class provides a unified interface for working with scikit-learn estimators.
    It allows dynamic instantiation of the estimator with default or custom hyperparameters,
    and exposes standard scikit-learn methods such as `fit`, `predict`, `get_params`, and
    `set_params`. This makes it easy to use any scikit-learn estimator in a consistent way
    within the framework.

    Attributes:
        estimator_class (Type[Any]): The scikit-learn estimator class to wrap, e.g.,
            `sklearn.linear_model.Ridge`.
        estimator (Optional[Any]): Instance of the fitted estimator. This is set after
            calling `fit`.
        name (str): Name of the estimator class, derived from `estimator_class.__name__`.
        hyperparameters (Dict[str, Any]): Dictionary of hyperparameter values used to
            instantiate the estimator.
        hyperparameters_space (Dict[str, List[Any]]): Dictionary defining the search
            space for hyperparameters for tuning or optimization.
        X_train_ (Optional[np.ndarray]): Optional attribute to store training data, if needed
            for reference or internal operations.
    """

    def __init__(self, estimator_class: Type[Any]):
        """
        Initialize the SklearnEstimator wrapper with a given estimator class.

        Args:
            estimator_class (Type[Any]): The scikit-learn estimator class to wrap.
        """
        self.estimator = None
        self.estimator_class = estimator_class
        self.name = self.estimator_class.__name__
        self.hyperparameters = get_default_parameters(estimator_class)
        self.hyperparameters_space = get_hyperparameters_space(self.name)
        self.X_train_: Optional[np.ndarray] = None

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
        self.X_train_ = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using the fitted Scikit-learn estimator.

        Args:
            X (np.ndarray): Input feature matrix for prediction.

        Raises:
            ValueError: If `fit()` has not been called before prediction.

        Returns:
            np.ndarray: Predicted target values.
        """
        if self.estimator is None:
            raise ValueError("You must call `fit()` before `predict()`.")
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

    def fit_cv(
        self,
        X: np.ndarray,
        cv: Union[int, _BaseKFold] = 5,
    ) -> BaseEstimator:
        """
        Perform a Grid Search over the estimator's hyperparameters
        and return a trained instance of the best model.

        Args:
            X (np.ndarray): Training feature matrix.
            cv (int or BaseCrossValidator): Cross-validation strategy. Default is 5-fold.

        Returns:
            BaseEstimator: An instance of the estimator with the best
            hyperparameters, already fitted to the training data.
        """
        # First of all, split data into two sets: feature and target.
        # To begin with, only one lag is taken into account.
        steps = X.shape[0]
        feautre = X[: (steps - 1), :]
        target = X[1:, :]

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
        grid.fit(feautre, target)

        # Instantiate and fit the best estimator
        self.hyperparameters = grid.best_params_
        self.fit(feautre, target)

        return self

    def in_sample_estimation(self, steps: int) -> np.ndarray:
        """
        Generate in sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Raises:
            ValueError: If `fit()` has not been called before in_sample_estimation.

        Returns:
            np.ndarray: Matrix containing the estimated values.
        """
        if self.X_train_ is None:
            raise ValueError("You must call `fit()` before `in_sample_estimation()`.")
        estimated_values = self.predict(self.X_train_[-steps:, :])
        return estimated_values

    def out_sample_estimation(self, steps: int) -> np.ndarray:
        """
        Generate out sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Raises:
            ValueError: If `fit()` has not been called before out_sample_estimation.

        Returns:
            np.ndarray: Matrix containing the estimated values.
        """
        if self.X_train_ is None:
            raise ValueError("You must call `fit()` before `in_sample_estimation()`.")
        estimated_values = np.full((steps, self.X_train_.shape[1]), np.nan)
        current_data = np.reshape(self.X_train_[-1, :], shape=(1, -1))
        for i_pred in range(steps):
            current_estimation = np.reshape(self.predict(current_data), shape=(1, -1))
            estimated_values[i_pred, :] = current_estimation
            current_data = current_estimation
        return estimated_values

    def get_score(self, X: np.ndarray) -> float:
        """
        Compute the score metric using the data provided.

        Args:
            X (np.ndarray): Matrix containing the features.
            estimator (EstimatorProtocol): The estimator to be evaluated.

        Raises:
            ValueError: If `fit()` has not been called before computing the score.
            ValueError: If `X` does not have the correct shape (at least 2 samples).

        Returns:
            float: The scoring measure.
        """
        if self.X_train_ is None:
            raise ValueError("You must call `fit()` before `get_score()`.")
        if len(X.shape) != 2 or (len(X.shape) == 2 and X.shape[0] < 2):
            raise ValueError("X must be of appropriate shape")
        estimated_values = self.predict(X)
        X_aligned = X[1:, :]
        estimated_aligned = estimated_values[:-1, :]
        return rmse(X_aligned, estimated_aligned)

    def estimate_residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate residuals.

        Args:
            X (np.ndarray): Input feature matrix for prediction. This
                matrix is used as the true values.

        Raises:
            ValueError: If `fit()` has not been called before computing the residuals.

        Returns:
            np.ndarray: Residuals.
        """
        if self.X_train_ is None:
            raise ValueError("You must call `fit()` before `estimate_residuals()`.")
        estimated_values = self.predict(X)
        X_true = X[1:, :]
        X_estimated = estimated_values[:-1, :]
        return X_true - X_estimated
