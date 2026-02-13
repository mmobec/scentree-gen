from inspect import signature
from typing import Any, Dict, List, Type


HYPERPARAMETERS_SPACE: Dict[str, Dict[str, List[Any]]] = {
    "Ridge": {
        "alpha": [0.05, 0.06, 0.08, 0.1, 0.2, 0.25, 0.5, 0.7, 1, 2],
        "max_iter": [100, 200, 300, 500, 1000],
    },
    "VAR": {"maxlags": [2, 5, 7, 10], "trend": ["n", "c", "ct", "ctt"]},
}


def get_hyperparameters_space(estimator_name: str) -> Dict[str, List[Any]]:
    """Obtain the hyperparameters space for a given estimator.

    Args:
        estimator_name (str): Name of the estimator.

    Returns:
        dict: Dictionary with the range for each hyperparameter.
    """
    return HYPERPARAMETERS_SPACE.get(estimator_name, {})


def get_default_parameters(class_chosen: Type, from_fit: bool = False) -> Dict[str, Any]:
    """
    Retrieve the default hyperparameters for a given estimator class.

    This function inspects the constructor (`__init__`) or the `fit` method
    of the specified estimator class and returns a dictionary containing
    all parameters that have default values, excluding `self`.

    Args:
        class_chosen (Type): The estimator class to retrieve default parameters from.
        from_fit (bool, optional): If True, retrieve defaults from the `fit` method
            instead of the constructor. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary mapping parameter names to their default values.
            Only parameters with default values are included.

    """

    target = class_chosen.fit if from_fit else class_chosen.__init__
    sig = signature(target)
    defaults = {
        k: v.default for k, v in sig.parameters.items() if v.default is not v.empty and k != "self"
    }
    return defaults
