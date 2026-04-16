import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scentree.config import explained_var
from scentree.dim_reduction.pca import BasePCA
from scentree.estimators.estimator_orchestrator import EstimatorController
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, TypedDict


class ScenariosPayload(TypedDict):
    """
    Container for scenario generation results.

    Attributes:
        scenarios (List[NDArray[np.float64]]): Generated scenarios.
        predicted_values (List[NDArray[np.float64]]): Predicted values associated
            with each scenario.
        observed_values Optional[List[NDArray[np.float64]]]: Observed values
            corresponding to each scenario, if available.
    """

    scenarios: List[NDArray[np.float64]]
    predicted_values: List[NDArray[np.float64]]
    observed_values: Optional[List[NDArray[np.float64]]]


class StageManager(BaseModel):
    """
    Manages the workflow of a stage.
    """

    model_config = {"arbitrary_types_allowed": True}

    def get_scenarios(
        self,
        residuals: NDArray[np.float64],
        estimated_values: NDArray[np.float64],
        num_trees: int,
        num_scenarios: int,
        seed: Optional[int] = None,
    ) -> List[NDArray[np.float64]]:
        """
        Obtain the scenarios given all components that are needed.

        Args:
            residuals(NDArray[np.float64]): Matrix containing historical residuals.
            estimated_values (NDArray[np.float64]): Estimated values.
            num_trees (int): Number of trees to provide.
            num_scenarios (int): Number of scenarios to generate the fan.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Returns:
            List[NDArray[np.float64]]: List containing the scenarios.
        """
        scenarios = []
        # For each day, generate the scenarios
        for i_day in range(num_trees):
            estimated_value = np.reshape(estimated_values[i_day, :], (1, -1))
            vector_ones = np.ones((num_scenarios, 1))
            # Transform it to a matrix. Each row is repeated
            estimated_matrix = np.matmul(vector_ones, estimated_value)
            # Take randonmly a sample of residuals
            rng = np.random.default_rng(seed)
            idx_residuals = rng.choice(a=residuals.shape[0], size=num_scenarios, replace=True)
            current_residuals = residuals[idx_residuals, :]
            current_scenarios = estimated_matrix + current_residuals
            scenarios.append(current_scenarios)
        return scenarios

    def generate_scenarios_payload(
        self,
        X: NDArray[np.float64],
        num_trees: int,
        num_scenarios: int,
        build_in_sample_trees: bool = True,
        seed: Optional[int] = None,
    ) -> ScenariosPayload:
        """
        Orchestrates the scenario generation process from historical data.

        Args:
            X (NDArray[np.float64]): Historical data.
            num_trees (int):  Number of trees to generate.
            num_scenarios (int): Number of scenarios to generate.
            build_in_sample_trees (bool): Whether to build in sample trees or
                out sample trees. Default to True, meaning that in sample trees are built.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Returns:
            ScenariosPayload:
                A container with:
                    - scenarios: generated scenarios
                    - predicted_values: model predictions
                    - observed_values: optional observed values
        """
        # Perform normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform dimensionality reduction
        dim_reduction = BasePCA()
        X_reduced = dim_reduction.fit_auto_components(X_scaled, threshold=explained_var)

        # Find the best estimator
        estimator_controller = EstimatorController()
        estimator_controller.fit(X=X_reduced)

        # Estimate the residuals
        residuals = estimator_controller.estimate_residuals(X_reduced)

        # Get estimated values
        if build_in_sample_trees:
            estimated_values = estimator_controller.in_sample_estimation(num_trees)
            observed_values = [row for row in X[-num_trees:, :]]
        else:
            estimated_values = estimator_controller.out_sample_estimation(num_trees)
            observed_values = None

        # Create scenarios for all num_trees
        scenarios = self.get_scenarios(
            residuals,
            estimated_values,
            num_trees,
            num_scenarios,
            seed,
        )
        # For each tree, recover the data in the high dimensional space
        scenarios_high = []
        estimated_values_high = dim_reduction.inverse_transform(estimated_values)
        estimated_original = scaler.inverse_transform(estimated_values_high)
        for current_scenarios in scenarios:
            sc_high = dim_reduction.inverse_transform(current_scenarios)
            sc_original = scaler.inverse_transform(sc_high)
            scenarios_high.append(sc_original)
        results: ScenariosPayload = {
            "scenarios": scenarios_high,
            "predicted_values": [row for row in estimated_original],
            "observed_values": observed_values,
        }
        return results
