import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scentree.config import explained_var
from scentree.dim_reduction.pca import BasePCA
from scentree.estimators.estimator_orchestrator import EstimatorController
from scentree.io.loader import Bounds
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, TypedDict


class ScenarioFanData(TypedDict):
    """
    Data container representing a scenario fan in a stochastic simulation framework.

    A scenario fan is a collection of independently generated scenario realizations
    derived from a common initial state. Each scenario represents a possible outcome
    of the underlying stochastic process.

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
    Orchestrates the generation of a scenario fan from historical data and model estimates.

    The StageManager is responsible for transforming historical time series data into
    a stochastic representation of future uncertainty. It performs preprocessing,
    dimensionality reduction, residual modeling, scenario generation, and reconstruction
    back to the original feature space.
    """

    model_config = {"arbitrary_types_allowed": True}

    def get_scenarios(
        self,
        residuals: NDArray[np.float64],
        estimated_values: NDArray[np.float64],
        num_fans: int,
        num_scenarios: int,
        seed: Optional[int] = None,
    ) -> List[NDArray[np.float64]]:
        """
        Obtain the scenarios given all components that are needed.

        Args:
            residuals(NDArray[np.float64]): Matrix containing historical residuals.
            estimated_values (NDArray[np.float64]): Estimated values.
            num_fans (int): Number of fans to provide.
            num_scenarios (int): Number of scenarios to generate the fan.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Returns:
            List[NDArray[np.float64]]: List containing the scenarios.
        """
        scenarios = []
        # For each day, generate the scenarios
        for i_day in range(num_fans):
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

    def clip_matrix_values(self, X: NDArray, value_ranges: Optional[List[Bounds]]) -> NDArray:
        """
        Clip the values of a given matrix.

        Args:
            X (NDArray): Matrix to be clipped.
            value_ranges (Optional[List[Bounds]]): Bounds used to clip X.
        """
        X_clipped = X.copy()
        if value_ranges is not None:
            for i, vr in enumerate(value_ranges):
                if vr is not None:
                    X_clipped[:, i] = np.clip(X[:, i], vr[0], vr[1])
        return X_clipped

    def generate_scenario_fans(
        self,
        X: NDArray[np.float64],
        num_fans: int,
        num_scenarios: int,
        stage_ids: Optional[List[int]] = None,
        num_variables_per_stage: Optional[List[int]] = None,
        initial_stochastic_stage_id: Optional[int] = None,
        build_in_sample_fans: bool = True,
        value_ranges: Optional[List[Bounds]] = None,
        seed: Optional[int] = None,
    ) -> ScenarioFanData:
        """
        Orchestrates the scenario fan generation process from historical data.

        Args:
            X (NDArray[np.float64]): Historical data.
            num_fans (int):  Number of fans to generate.
            num_scenarios (int): Number of scenarios to generate.
            stage_ids (Optional[List[int]]): Stages of the stochastic problem.
            num_variables_per_stage (Optional[List[int]]): number of random variables involved
                in each stage.
            initial_stochastic_stage_id (Optional[List[int]]): First stage that is random.
                Previous stages are considered observed, i.e., fixed.
            build_in_sample_fans (bool): Whether to build in-sample fans or
                out-sample fans. Default to True, meaning that in sample fans are built.
            value_ranges (Optional[List[Bounds]]): Optional lower and upper bounds
                for each variable in `X`.
            seed (Optional[int]): Seed needed in case reproducibility is required.

        Raises:
            ValueError:
                - If the length of `value_ranges` does not match the number of columns in `X`.
                - If a value range is invalid, i.e. the lower bound is greater than the upper bound.
                - If `initial_stochastic_stage_id` is provided but `stage_ids` is `None`.
                - If `initial_stochastic_stage_id` is not present in `stage_ids`.
                - If `initial_stochastic_stage_id` is provided but `num_variables_per_stage` is
                    `None`.
                - If the length of `num_variables_per_stage` does not match the length of
                    `stage_ids`.

        Returns:
            ScenarioFanData:
                A container with:
                    - scenarios: generated scenarios
                    - predicted_values: model predictions
                    - observed_values: optional observed values
        """
        if value_ranges is not None:
            if len(value_ranges) != X.shape[1]:
                raise ValueError(
                    "The length of `value_ranges` must be equal to the number of columns of `X`"
                )
            for i, rv in enumerate(value_ranges):
                if rv is not None and rv[0] > rv[1]:
                    raise ValueError(
                        f"""The first value of `value_ranges` must be greater than the
                        second value in position {i}"""
                    )
        if initial_stochastic_stage_id is not None:
            if stage_ids is None:
                raise ValueError(
                    """If `initial_stochastic_stage_id` is provided
                    then `stage_ids` must be provided"""
                )
            else:
                if initial_stochastic_stage_id not in stage_ids:
                    raise ValueError(
                        """`initial_stochastic_stage_id` must be a value
                        of the list `stage_ids`"""
                    )
            if num_variables_per_stage is None:
                raise ValueError(
                    """If `initial_stochastic_stage_id` is provided
                    then `num_variables_per_stage` must be provided"""
                )
            else:
                if len(num_variables_per_stage) != len(stage_ids):
                    raise ValueError(
                        """`num_variables_per_stage` and `stage_ids` must be of
                        the same length"""
                    )

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
        if build_in_sample_fans:
            estimated_values = estimator_controller.in_sample_estimation(num_fans)
            observed_values = [row for row in X[-num_fans:, :]]
        else:
            estimated_values = estimator_controller.out_sample_estimation(num_fans)
            observed_values = None

        # Create scenarios for all num_fans
        scenarios_low = self.get_scenarios(
            residuals,
            estimated_values,
            num_fans,
            num_scenarios,
            seed,
        )
        # For each fan, recover the data in the high dimensional space
        scenarios_high = []
        estimated_values_high = dim_reduction.inverse_transform(estimated_values)
        estimated_original = scaler.inverse_transform(estimated_values_high)
        for current_scenarios in scenarios_low:
            sc_high = dim_reduction.inverse_transform(current_scenarios)
            sc_original = scaler.inverse_transform(sc_high)
            sc_original_clipped = self.clip_matrix_values(sc_original, value_ranges)
            scenarios_high.append(sc_original_clipped)
        predicted_values = [row for row in estimated_original]
        if (
            initial_stochastic_stage_id is not None
            and stage_ids is not None
            and num_variables_per_stage is not None
        ):
            scenarios = self.fill_with_constant_values(
                initial_stochastic_stage_id,
                scenarios_high,
                predicted_values,
                observed_values,
                stage_ids,
                build_in_sample_fans,
                num_variables_per_stage,
            )
        else:
            scenarios = scenarios_high
        results: ScenarioFanData = {
            "scenarios": scenarios,
            "predicted_values": predicted_values,
            "observed_values": observed_values,
        }
        return results

    def fill_with_constant_values(
        self,
        initial_stochastic_stage_id: int,
        scenarios: List[NDArray[np.float64]],
        predicted_values: List[NDArray[np.float64]],
        observed_values: Optional[List[NDArray[np.float64]]],
        stage_ids: List[int],
        build_in_sample_fans: bool,
        num_variables_per_stage: List[int],
    ) -> List[NDArray[np.float64]]:
        """
        Fill the deterministic stages of the generated scenarios with constant values
        derived from either observed or predicted values.

        For all stages before `initial_stochastic_stage_id`, the values are fixed across
        all scenarios in the fan.

        Args:
            initial_stochastic_stage_id (int): Identifier of the first stochastic stage.
            scenarios (List[NDArray[np.float64]]): Generated scenarios for each fan.
            predicted_values (List[NDArray[np.float64]]): Predicted values associated with each fan.
            observed_values (Optional[List[NDArray[np.float64]]]): Observed values associated
                with each fan. Used only when `build_in_sample_fans` is `True`.
            stage_ids (List[int]): Ordered list of stage identifiers.
            build_in_sample_fans (bool): Whether to use observed values instead of predicted values.
            num_variables_per_stage (List[int]): Number of variables associated with each stage.

        Returns:
            List[NDArray[np.float64]]: Scenarios where all variables belonging to stages before
                `initial_stochastic_stage_id` are fixed to constant values.
        """
        results = []
        if build_in_sample_fans and observed_values is not None:
            values = observed_values.copy()
        else:
            values = predicted_values.copy()
        # Identify the position of the first stochastic stage_id
        idx = stage_ids.index(initial_stochastic_stage_id)
        total_rv_fixed = sum(num_variables_per_stage[:idx])
        total_fans = len(values)
        scenario_shape = scenarios[0].shape
        for i in range(total_fans):
            current_value = values[i]
            current_scenarios = scenarios[i]
            current_value_r = np.reshape(current_value, (1, -1))
            ones_vector = np.reshape(np.ones(scenario_shape[0]), (-1, 1))
            matrix_current_value = np.matmul(ones_vector, current_value_r[:, :total_rv_fixed])
            # Fill the sceanrios with the fixed values
            current_scenario_fixed = current_scenarios.copy()
            current_scenario_fixed[:, :total_rv_fixed] = matrix_current_value
            results.append(current_scenario_fixed)
        return results
