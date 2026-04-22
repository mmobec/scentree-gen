import json
import logging
import numpy as np
from datetime import datetime
from numpy.typing import NDArray
from pathlib import Path
from scentree.io import MapColsNamesStages
from scentree.tree_construction import Tree
from typing import List, Union

logger = logging.getLogger(__name__)


def save_json(
    output_dir: Union[str, Path],
    num_stages: int,
    in_sample_prediction: bool,
    predicted_value: List[NDArray],
    observed_value: Union[List[NDArray], None],
    scenarios: List[NDArray[np.float64]],
    trees: List[Tree],
    mapping_datasets_columns: MapColsNamesStages,
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
        scenarios (List[NDArray[np.float64]]): List of scenario matrices,
            one per tree.
        trees (List[Tree]): List of tree structures corresponding to each
            scenario set.
        mapping_datasets_columns (MapColsNamesStages): Mapping between dataset names,
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
    for i in range(len(trees)):
        current_scenarios = scenarios[i]
        current_predicted_value = predicted_value[i].tolist()
        if observed_value is not None:
            current_observed_value = observed_value[i].tolist()
        else:
            current_observed_value = None
        scenario_probabilities, nodes = trees[i]
        information = {
            "num_scenarios": current_scenarios.shape[0],
            "num_stages": num_stages,
            "in_sample_prediction": in_sample_prediction,
            "scenarios": current_scenarios.tolist(),
            "mean_scenarios": np.mean(current_scenarios, axis=0).tolist(),
            "predicted_value": current_predicted_value,
            "observed_value": current_observed_value,
            "scenario_probabilities": scenario_probabilities.tolist(),
            "mapping_datasets_columns": mapping_datasets_columns,
            "tree": nodes,
        }
        data.append(information)
    if multiple_files:
        pass
    else:
        with open(new_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
    logger.info(f"Results saved in {new_dir}")
    return None
