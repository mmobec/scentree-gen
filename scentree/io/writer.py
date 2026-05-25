import json
import logging
import numpy as np
from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path
from scentree.io import DatasetMappings
from scentree.tree_construction import ScenarioTrees
from typing import List, Union

logger = logging.getLogger(__name__)


def save_json(
    output_dir: Union[str, Path],
    num_stages: int,
    in_sample_prediction: bool,
    predicted_value: List[NDArray],
    observed_value: Union[List[NDArray], None],
    scenario_trees: ScenarioTrees,
    mapping_datasets_columns: DatasetMappings,
    multiple_files: bool = False,
) -> None:
    """
    Save model results and scenario trees into a JSON file.

    This function stores simulation results, predictions, and tree structures
    into a timestamped output directory. It supports both single-file and
    multi-file export modes.

    Args:
        output_dir (Union[str, Path]): Directory where results will be saved.
        num_stages (int): Number of stages in the scenario model.
        in_sample_prediction (bool): Whether predictions are in-sample.
        predicted_value (List[NDArray]): List of predicted value arrays,
            one per tree.
        observed_value (Union[List[NDArray], None]): List of observed values,
            or None if not available.
        scenario_trees (ScenarioTrees): List of scenario tree. Each position contains a
            scenario tree.
        mapping_datasets_columns (DatasetMappings): Mapping between dataset names,
            their column indices and their stages.
        multiple_files (bool): If True, results are saved in separate files
            (not yet implemented). If False, all results are stored in a
            single JSON file.

    Raises:
        FileNotFoundError: If `output_dir` does not exist.
        NotADirectoryError: If `output_dir` is not a directory.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Directory {output_dir} does not exist")

    if not output_dir.is_dir():
        raise NotADirectoryError(f"{output_dir} is not a directory")

    base_name = datetime.now().strftime("results_%Y%m%d_%H%M%S")
    new_dir = output_dir / base_name
    counter = 1
    while new_dir.exists():
        new_dir = output_dir / f"{base_name}_{counter}"
        counter += 1
    new_dir.mkdir()
    data = []
    for i in range(len(scenario_trees)):
        current_scenario_tree = scenario_trees[i]
        current_data = current_scenario_tree["scenario_tree_data"]
        current_tree = current_scenario_tree["tree"]
        current_predicted_value = predicted_value[i]
        if observed_value is not None:
            current_observed_value = observed_value[i]
        else:
            current_observed_value = None
        scenario_probabilities = current_scenario_tree["scenario_probabilities"]
        mean_value_scenario_tree = np.dot(scenario_probabilities, current_data)
        information = {
            "num_scenarios": current_data.shape[0],
            "num_stages": num_stages,
            "in_sample_prediction": in_sample_prediction,
            "scenario_tree_data": current_data.tolist(),
            "mean_value_scenario_tree": mean_value_scenario_tree.tolist(),
            "predicted_value": current_predicted_value.tolist(),
            "observed_value": current_observed_value.tolist()
            if current_observed_value is not None
            else current_observed_value,
            "scenario_probabilities": scenario_probabilities.tolist(),
            "mapping_datasets_columns": mapping_datasets_columns,
            "tree": current_tree,
        }
        data.append(information)
    if multiple_files:
        for i, info in enumerate(data):
            info_list = [info]
            with open(new_dir / f"results_{i + 1}.json", "w", encoding="utf-8") as f:
                json.dump(info_list, f, indent=1)
    else:
        with open(new_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    logger.info(f"Results saved in {new_dir}")
    return None
