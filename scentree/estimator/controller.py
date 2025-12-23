import numpy as np
from copy import copy
from pydantic import BaseModel
from scentree.estimator.ridge import RidgeEstimator
from sklearn.model_selection._split import _BaseKFold
from scentree.metrics.rmse import rmse
from typing import List, Type, Union


class EstimatorController(BaseModel):
    """Controller class for managing and selecting estimators.

    This class provides a mechanism to manage multiple estimator types
    within the scentree framework. It trains each estimator using the
    provided data, evaluates their performance based on the scoring measure.

    Attributes:
        estimator_classes (List[Type]): A list of estimator classes to be
            evaluated.
    """

    estimator_classes: List[Type] = [RidgeEstimator]

    def get_score(self, X: np.ndarray, y: np.ndarray, estimator: RidgeEstimator) -> float:
        """
        Compute the score metric using the data provided.

        Args:
            X (np.ndarray): Input feature matrix for prediction.
            y (np.ndarray): Input target values.
            estimator (RidgeEstimator): The estimator to be evaluated.

        Returns:
            float: the scoring measure.
        """
        prediction = estimator.predict(X)
        score = rmse(y, prediction)
        return score

    def get_best_estimator(
        self, X: np.ndarray, y: np.ndarray, cv: Union[int, _BaseKFold] = 5
    ) -> RidgeEstimator:
        """Train and select the best estimator based on R² score.

        This method iterates over all estimator classes defined in
        `self.estimator_classes`. Each estimator is trained using its
        `get_model` method, evaluated on the data using the metric,
        and compared to find the best-performing model. The
        estimator achieving the highest score is returned.

        Args:
            X (np.ndarray): Input feature matrix for prediction.
            y (np.ndarray): Input target values.
            cv (Union[int, _BaseKFold], optional): Cross-validation configuration
                passed to the estimator's `get_model` method. Defaults to 5.

        Returns:
            RidgeEstimator: The fitted estimator with the highest R² score
            on the training data.

        Raises:
            ValueError: If no estimator is selected (i.e., `estimator_chosen` is None).
        """
        best_score = None
        estimator_chosen = None
        for EstimatorClass in self.estimator_classes:
            # Instantiate the estimator
            estimator = EstimatorClass()
            best_estimator = estimator.get_model(X=X, y=y, cv=cv)
            # Compute score
            score = self.get_score(X=X, y=y, estimator=best_estimator)
            if (best_score is None) or (best_score is not None and score < best_score):
                estimator_chosen = copy(best_estimator)
                best_score = copy(score)
        # Evaluate the estimator in the test set
        if estimator_chosen is None:
            raise ValueError("`estimator_chosen` is None")
        return estimator_chosen
