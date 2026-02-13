import copy
import numpy as np
import itertools
from scentree.estimator.utils import get_default_parameters, get_hyperparameters_space
from scentree.metrics.rmse import rmse
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from typing import Any, Dict, Optional, TypeVar, Union


R = TypeVar("R", bound="VarEstimator")


class VarEstimator:
    """
    Configuration and state variables for the VAR estimator wrapper.

    Attributes:
        estimator_class (Type): The estimator class being used, here VAR from statsmodels.
        name (str): Name of the estimator class, derived from `estimator_class.__name__`.
        hyperparameters (Dict[str, Any]): Default hyperparameters for the estimator, obtained
            from the `fit` method using `get_default_parameters`.
        hyperparameters_space (Dict[str, List[Any]]): Search space for hyperparameters,
            used for hyperparameter optimization or tuning.
        results (Optional[VARResultsWrapper]): Storage for the results of the fitted VAR model.
            Initially None, set after fitting.
        X_train_ (Optional[np.ndarray]): Training data used for fitting the model.
            Stored for reference or further computation.
    """

    estimator_class = VAR
    name = VAR.__name__
    hyperparameters = get_default_parameters(estimator_class, from_fit=True)
    hyperparameters_space = get_hyperparameters_space(name)
    results: Optional[VARResultsWrapper] = None
    X_train_: Optional[np.ndarray] = None

    model_config = {"arbitrary_types_allowed": True}

    def fit(self: R, X: np.ndarray) -> R:
        """Fit the statsmodels estimator to the training data.

        Args:
            X (np.ndarray): Input data for fitting the model.

        Raises:
            ValueError: If hypermaters have not been provided previously.
            ValueError: If the estimator has not been provided previously.

        Returns:
            R: The fitted wrapper instance (same type as self).
        """
        if self.hyperparameters is None:
            raise ValueError("Hyperparameters must be initialized before fitting.")
        self.estimator = self.estimator_class(endog=X)
        fit_params = dict(self.hyperparameters)
        # Ensure maxlags does not exceed number of observations
        if (
            "maxlags" in fit_params
            and fit_params["maxlags"] is not None
            and X.shape[0] <= fit_params["maxlags"]
        ):
            fit_params["maxlags"] = X.shape[0] - 1
        if self.estimator is None:
            raise ValueError("Estimator must be initialized before fitting.")
        self.X_train_ = X.copy()
        self.results = self.estimator.fit(**fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts using the fitted statsmodels estimator.
        Args:
            X (np.ndarray): Input data for which to make predictions; its shape
                determines the number of forecast steps.

        Raises:
            ValueError: If 'fit()' has not been called before prediction.

        Returns:
            np.ndarray: Forecasted values.
        """
        steps = X.shape[0]
        if self.results is None or self.X_train_ is None:
            raise ValueError("You must call `fit()` before `predict()`.")
        p = self.results.k_ar
        forecasted_values = self.results.forecast(y=self.X_train_[-p:, :], steps=steps)
        return forecasted_values

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
        if self.results is None:
            raise ValueError("You must call `fit()` before `in_sample_estimation()`.")
        fitted = self.results.fittedvalues
        if steps > fitted.shape[0]:
            raise ValueError("`steps` cannot exceed the number of in-sample predictions.")
        return fitted[-steps:, :]

    def out_sample_estimation(self, steps: int) -> np.ndarray:
        """
        Generate out sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Raises:
            ValueError: If `fit()` has not been called before out_sample_estimation.
            ValueError: If `steps` exceeds the number of samples.

        Returns:
            np.ndarray: Matrix containing the estimated values.
        """
        if self.results is None or self.X_train_ is None:
            raise ValueError("You must call `fit()` before `out_sample_estimation()`.")
        p = self.results.k_ar
        forecasted_values = self.results.forecast(y=self.X_train_[-p:, :], steps=steps)
        return forecasted_values

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the parameters of the wrapper.

        Args:
            deep (bool): Ignored; included for compatibility with scikit-learn API.

        Returns:
            Dict[str, Any]: Dictionary containing 'estimator_class',
            'hyperparameters', and 'hyperparameters_space'.
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
        if self.hyperparameters is None:
            raise ValueError("Hyperparameters must be initialized before using.")
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
        self: R,
        X: np.ndarray,
        cv: Union[int, BaseCrossValidator] = 5,
    ) -> R:
        """Perform cross-validation over the hyperparameter space and fit the best model.

        Args:
            X (np.ndarray): Input feature matrix.
            cv (Union[int, BaseCrossValidator]): Number of CV splits or a cross-validator instance.

        Returns:
            R: The fitted wrapper instance with the best hyperparameters (same type as self)..
        """
        if self.hyperparameters_space is None:
            raise ValueError("hyperparameters_space must be initialized before performing CV.")

        if isinstance(cv, int):
            cv_splitter = TimeSeriesSplit(n_splits=cv)
        else:
            cv_splitter = cv

        best_score = np.inf
        best_params = None
        param_names = list(self.hyperparameters_space.keys())
        param_values = [self.hyperparameters_space[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))

        for combo in param_combinations:
            current_params = dict(zip(param_names, combo))
            cv_scores = []
            i = 1
            for train_idx, test_idx in cv_splitter.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                fold_estimator = copy.deepcopy(self)
                fold_estimator.set_params(**current_params)
                fold_estimator.fit(X_train)
                if fold_estimator.results is None:
                    raise ValueError("You must call `fit()`")
                score = fold_estimator.get_score(X_test)
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_params = current_params
            i += 1
        if best_params is None:
            raise ValueError("No best_params found. Check your hyperparameter space.")
        self.set_params(**best_params)
        self.fit(X)
        return self

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
        if self.results is None:
            raise ValueError("You must call `fit()` before `estimate_residuals()`.")
        p = self.results.k_ar
        X_true = X[p:, :]
        X_estimated = self.results.fittedvalues
        residuals = X_true - X_estimated
        return residuals

    def get_score(self, X: np.ndarray) -> float:
        """
        Compute the score metric using the data provided.

        Args:
            X (np.ndarray): Matrix containing the features.
            estimator (EstimatorProtocol): The estimator to be evaluated.

        Raises:
            ValueError: If `fit()` has not been called before computing the score.

        Returns:
            float: The scoring measure.
        """
        if self.results is None:
            raise ValueError("You must call `fit()` before `get_score()`.")
        estimated_values = self.predict(X)
        score = rmse(X, estimated_values)
        return score
