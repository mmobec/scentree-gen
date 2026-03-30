import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scentree.estimators.ridge import RidgeEstimator
from scentree.estimators.var import VarEstimator
from sklearn.model_selection import BaseCrossValidator
from typing import Any, cast, ClassVar, List, Optional, Protocol, Self, Type, TypeVar, Union

R = TypeVar("R", bound="EstimatorController")


class EstimatorProtocol(Protocol):
    """
    Protocol defining the minimal interface required for an estimator.

    This protocol standardizes the expected behavior for models that support
    prediction, residual estimation, cross-validation fitting, in-sample and
    out-of-sample estimation, and scoring.
    """

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate predictions.

        Args:
            X (NDArray[np.float64]): Input feature matrix for prediction.

        Returns:
            NDArray[np.float64]: Predicted target values.
        """
        ...

    def estimate_residuals(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Estimate residuals.

        Args:
            X (NDArray[np.float64]): Input feature matrix for prediction. This
                matrix is used as the true values.

        Returns:
            NDArray[np.float64]: Residuals.
        """
        ...

    def fit_cv(
        self,
        X: NDArray[np.float64],
        y: Optional[NDArray[np.float64]],
        cv: Union[int, BaseCrossValidator],
    ) -> Self:
        """
        Fit an estmator using cross-validation.

        Args:
            X (NDArray[np.float64]): Input feature matrix.
            y (Optional[NDArray[np.float64]]): Target value.
            cv (Union[int, BaseCrossValidator]): Number of folds (if int) or a
                scikit-learn compatible cross-validator instance defining
                the splitting strategy.

        Returns:
            Self: The fitted estimator instance.
        """
        ...

    def in_sample_estimation(self, steps: int) -> NDArray[np.float64]:
        """
        Generate in sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Returns:
            NDArray[np.float64]: Matrix containing the estimated values.
        """
        ...

    def out_sample_estimation(self, steps: int) -> NDArray[np.float64]:
        """
        Generate out sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Returns:
            NDArray[np.float64]: Matrix containing the estimated values.
        """
        ...

    def get_score(self, X: NDArray[np.float64]) -> float:
        """
        Compute the score metric using the data provided.

        Args:
            X (NDArray[np.float64]): Matrix containing the features.

        Returns:
            float: The scoring measure.
        """
        ...


class EstimatorController(BaseModel):
    """Controller class for managing and selecting estimators.

    This class provides a mechanism to manage multiple estimator types.
    It trains each estimator using the provided data, evaluates their
    performance based on the scoring measure.

    Attributes:
        estimator_classes (ClassVar[List[Type]]): A list of estimator classes to be
            evaluated.
        best_estimator (Optional[Any]): The best estimator.
    """

    estimator_classes: ClassVar[List[Type]] = [RidgeEstimator, VarEstimator]
    best_estimator: Optional[Any] = None

    def get_score(self, X: NDArray[np.float64], estimator: EstimatorProtocol) -> float:
        """
        Compute the score metric using the data provided.

        Args:
            X (NDArray[np.float64]): Matrix containing the features.
            estimator (EstimatorProtocol): The estimator to be evaluated.

        Returns:
            float: The scoring measure.
        """
        score = estimator.get_score(X)
        return score

    def fit(
        self: R,
        X: NDArray[np.float64],
        cv: Union[int, BaseCrossValidator] = 5,
    ) -> R:
        """Train an estimator.

        Args:
            X (NDArray[np.float64]): Input feature matrix for prediction.
            cv (Union[int, BaseCrossValidator]): Cross-validation configuration.
                Defaults to 5.

        Raises:
            ValueError: If no estimator is selected (i.e., `estimator_chosen` is None).

        Returns:
            EstimatorController: The fitted estimator.
        """
        best_score = None
        estimator_chosen = None
        for EstimatorClass in self.estimator_classes:
            # Instantiate the estimator
            estimator = EstimatorClass()
            estimator.fit_cv(X=X, cv=cv)
            # Compute score
            score = self.get_score(X=X, estimator=estimator)
            if (best_score is None) or (best_score is not None and score < best_score):
                estimator_chosen = estimator
                best_score = score
        # Evaluate the estimator in the test set
        if estimator_chosen is None:
            raise ValueError("`estimator_chosen` is None")
        self.best_estimator = estimator_chosen
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate predictions.

        Args:
            X (NDArray[np.float64]): Input feature matrix for prediction.

        Raises:
            ValueError: If the estimator has not been previously fitted
                (i.e., `self.best_estimator` is None).

        Returns:
            NDArray[np.float64]: Predicted target values.
        """
        if self.best_estimator is None:
            raise ValueError("You must call `fit()` before `predict()`.")
        return cast(NDArray[np.float64], self.best_estimator.predict(X))

    def estimate_residuals(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Estimate residuals.

        Args:
            X (NDArray[np.float64]): Input feature matrix for prediction. This
                matrix is used as the true values.

        Raises:
            ValueError: If the estimator has not been previously fitted
                (i.e., `self.best_estimator` is None).

        Returns:
            NDArray[np.float64]: Residuals.
        """
        if self.best_estimator is None:
            raise ValueError("You must call `fit()` before `estimate_residuals()`.")
        return cast(NDArray[np.float64], self.best_estimator.estimate_residuals(X))

    def in_sample_estimation(self, steps: int) -> NDArray[np.float64]:
        """
        Generate in sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Raises:
            ValueError: If the estimator has not been previously fitted
                (i.e., `self.best_estimator` is None).

        Returns:
            NDArray[np.float64]: Matrix containing the estimated values.
        """
        if self.best_estimator is None:
            raise ValueError("You must call `fit()` before `in_sample_estimation()`.")
        return cast(NDArray[np.float64], self.best_estimator.in_sample_estimation(steps))

    def out_sample_estimation(self, steps: int) -> NDArray[np.float64]:
        """
        Generate out sample estimations.

        Args:
            steps (int): Number of estimated values to provide.

        Raises:
            ValueError: If the estimator has not been previously fitted
                (i.e., `self.best_estimator` is None).

        Returns:
            NDArray[np.float64]: Matrix containing the estimated values.
        """
        if self.best_estimator is None:
            raise ValueError("You must call `fit()` before `out_sample_estimation()`.")
        return cast(NDArray[np.float64], self.best_estimator.out_sample_estimation(steps))
