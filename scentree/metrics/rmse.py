import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Union

def rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Computes the root mean squared error.

    Args:
        y_true (np.ndarray): Real value.
        y_pred (np.ndarray): Predicted value.

    Returns:
        float: Root mean squared error.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))
