import copy
import itertools
import numpy as np
from pydantic import BaseModel, Field
from scentree.estimator.utils import get_default_parameters, get_hyperparameters_space
from scentree.metrics.rmse import rmse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union


R = TypeVar("R", bound="StatsEstimator")

class StatsEstimator(BaseModel):

    estimator_class: ClassVar[Type[Any]]
    estimator: Optional[Any] = None
    name: Optional[str] = None
    results: Optional[Any] = None
    X_train_: Optional[np.ndarray] = None
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    hyperparameters_space: Optional[Dict[str, List[Any]]] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": False,
    }

    def model_post_init(self, __context: Any) -> None:
        self.name = self.estimator_class.__name__
        if not self.hyperparameters:
            self.hyperparameters = get_default_parameters(self.estimator_class)
        if not self.hyperparameters_space:
            self.hyperparameters_space = get_hyperparameters_space(self.name)

    def fit(self: R, X: np.ndarray, y: Optional[np.ndarray] = None) -> R:
        if self.hyperparameters is None:
            raise ValueError("Hyperparameters must be initialized before fitting.")
        self.estimator = self.estimator_class(endog=X)
        fit_params = dict(self.hyperparameters)
        if 'maxlags' in fit_params and X.shape[0] <= fit_params['maxlags']:
            fit_params['maxlags'] = X.shape[0] - 1
        if self.estimator is None:
            raise ValueError("Estimator must be initialized before fitting.")
        self.X_train_ = X.copy()
        self.results = self.estimator.fit(**fit_params)
        return self

    # def predict(self, X: np.ndarray) -> np.ndarray:
    #     if self.results is None or self.X_train_ is None:
    #         raise ValueError("You must call `fit()` before `predict()`.")
    #     p = self.results.k_ar
    #     steps = X.shape[0]
    #     forecasted_values: np.ndarray = self.results.forecast(y=self.X_train_[-p:, :], steps=steps)
    #     return forecasted_values

    # def get_params(self, deep: bool = True) -> Dict[str, Any]:
    #     return {
    #         "estimator_class": self.estimator_class,
    #         "hyperparameters": self.hyperparameters,
    #         "hyperparameters_space": self.hyperparameters_space,
    #     }

    # def set_params(self: R, **params) -> R:
    #     """Set hyperparameter values for the estimator.

    #     Args:
    #         **params: Arbitrary keyword arguments of hyperparameters to update.

    #     Returns:
    #         SklearnEstimator: The updated estimator instance.
    #     """
    #     if self.hyperparameters is None:
    #         raise ValueError("Hyperparameters must be initialized before using.")
    #     for key, value in params.items():
    #         if hasattr(self, key):
    #             setattr(self, key, value)
    #         elif key in self.hyperparameters:
    #             self.hyperparameters[key] = value
    #         else:
    #             raise ValueError(
    #                 f"Invalid parameter '{key}' for {self.__class__.__name__}. "
    #                 f"Valid parameters: constructor={list(self.__dict__.keys())}, "
    #                 f"hyperparameters={list(self.hyperparameters.keys())}."
    #             )
    #     return self

    # def get_model(
    #     self: R,
    #     X: np.ndarray,
    #     y: Optional[np.ndarray] = None,
    #     cv: Union[int, _BaseKFold] = 5,
    # ) -> R:
    #     if self.hyperparameters_space is None:
    #         raise ValueError("hyperparameters_space must be initialized before performing CV.")

    #     if isinstance(cv, int):
    #         cv_splitter = TimeSeriesSplit(n_splits=cv)
    #     else:
    #         cv_splitter = cv

    #     best_score = np.inf
    #     best_params = None
    #     param_names = list(self.hyperparameters_space.keys())
    #     param_values = [self.hyperparameters_space[name] for name in param_names]
    #     param_combinations = list(itertools.product(*param_values))
        
    #     for combo in param_combinations:
    #         current_params = dict(zip(param_names, combo))
    #         cv_scores = []
    #         for train_idx, test_idx in cv_splitter.split(X):
    #             X_train, X_test = X[train_idx], X[test_idx]
    #             fold_estimator = copy.deepcopy(self)
    #             fold_estimator.set_params(**current_params)
    #             fold_estimator.fit(X_train)
    #             if fold_estimator.results is None:
    #                 raise ValueError("You must call `fit()`")
    #             forecast = fold_estimator.predict(X_test)
    #             score = rmse(X_test, forecast)
    #             cv_scores.append(score)
    #         mean_score = np.mean(cv_scores)
    #         if mean_score < best_score:
    #             best_score = mean_score
    #             best_params = current_params
    #     if best_params is None:
    #         raise ValueError("No best_params found. Check your hyperparameter space.")
    #     self.set_params(**best_params)
    #     self.fit(X)
    #     return self
