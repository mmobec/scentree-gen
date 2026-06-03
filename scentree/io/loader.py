import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator
from scentree.io import DatasetMappings
from typing import Dict, List, Optional, Self, Tuple


# Typing representing a range (a, b) where a is less than b
Bounds = Optional[Tuple[float, float]]
StageBounds = Optional[List[List[Bounds]]]
FullBouds = Optional[List[Bounds]]


class Dataset(BaseModel):
    """
    Container for the minimal information required to describe a dataset.

    Attributes:
        name (str): The name of the dataset.
        values (NDArray[np.float64]): The matrix containing the data.
        stage_ids (List[int]): The stage each column belongs to.
        bounds (Bounds): Optional tuple defining the lower and upper bounds of `values`.
    """

    name: str
    values: NDArray[np.float64]
    stage_ids: List[int]
    bounds: Bounds = None

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """
        Validate the internal consistency of the dataset specification.

        This validator ensures that the metadata describing the dataset
        (`stage_ids` and `bounds`) is consistent with the structure of
        `values`.

        The following conditions must hold:
            - The length of `stage_ids` must match the number of columns in `values`.
            - If `bounds` is not sorted, i.e., first value must be less than
                second value.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If any of the consistency conditions are violated.
        """
        if len(self.stage_ids) != self.values.shape[1]:
            raise ValueError(
                "The length of `stage_ids` must be equal to the number of columns in `values`"
            )
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError("`bounds[0]` must be less than or equal to bounds[1]`")
        return self


class DatasetsLoader(BaseModel):
    """
    Container that represents a collection of datasets.

    Attributes:
        datasets (List[Dataset]): Collection of datasets.
    """

    datasets: List[Dataset]

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_datasets_values(self) -> Self:
        """
        Validate that all datasets contain the same number of rows.

        This validator checks that the `values` matrix of every dataset in
        `datasets_information` has the same number of rows. The number of rows
        represents the number of observations, which must be consistent across
        all datasets.

        Raises:
            ValueError: If any dataset contains a different number of rows than
                the other datasets.

        Returns:
            Self: The validated model instance.
        """
        reference_idx = 0
        reference_n = self.datasets[reference_idx].values.shape[0]
        for i, dataset in enumerate(self.datasets):
            current_n = dataset.values.shape[0]
            if reference_n != current_n:
                raise ValueError(
                    f"""Dataset in position {reference_idx} contains {reference_n} rows.
                        Dataset in position {i} contains {current_n} rows.
                        All datasets must contain the same number of rows"""
                )
        return self

    def get_sorted_stage_ids(self) -> List[int]:
        """
        Retrieve all unique stage identifiers sorted in ascending order.

        Returns:
            List[int]: Sorted list of unique stage identifiers across all datasets.
        """
        unique_stage_ids = []
        for dataset in self.datasets:
            ds_stage_ids = dataset.stage_ids
            for stage_id in ds_stage_ids:
                if stage_id not in unique_stage_ids:
                    unique_stage_ids.append(stage_id)
        unique_stage_ids.sort()
        return unique_stage_ids

    def get_stage_values(self) -> List[NDArray[np.float64]]:
        """
        Construct the data matrices associated with each stage.

        Returns:
            List[NDArray[np.float64]]: List of stage matrices ordered according
                to the sorted stage identifiers. Each matrix contains all
                variables associated with the corresponding stage.
        """
        results = []
        unique_stage_ids = self.get_sorted_stage_ids()
        # Iterate over all sorted stages
        # Take all data for that stage for all datasets.
        # Append the columns
        for stage_id in unique_stage_ids:
            stage_values = []
            for ds in self.datasets:
                ds_stage_ids = ds.stage_ids
                filtered_idx = [i for i, st_id in enumerate(ds_stage_ids) if st_id == stage_id]
                filtered_ds_values = ds.values[:, filtered_idx]
                if filtered_ds_values.shape[0] > 0:
                    stage_values.append(filtered_ds_values)
            results.append(np.concatenate(stage_values, axis=1))
        return results

    def get_full_values(self) -> NDArray[np.float64]:
        """
        Construct the full data matrix by concatenating all stage matrices.

        Returns:
            NDArray[np.float64]: Full data matrix containing all variables
                across all stages. Rows correspond to observations and columns
                correspond to stage-ordered variables.
        """
        stage_values = self.get_stage_values()
        return np.concatenate(stage_values, axis=1)

    def get_stage_bounds(self) -> StageBounds:
        """
        Construct the bounds associated with each stage.

        Returns:
            StageBounds: Nested list containing the bounds associated with
                each variable of each stage. The outer list corresponds to
                stages, while the inner lists correspond to variables within
                the stage.

                Returns None if no dataset defines bounds.
        """
        results: StageBounds = []
        ds_stage_bounds = []
        unique_stage_ids = self.get_sorted_stage_ids()
        all_none = True
        for stage_id in unique_stage_ids:
            stage_bounds = []
            for ds in self.datasets:
                ds_stage_ids = ds.stage_ids
                filtered_idx = [i for i, st_id in enumerate(ds_stage_ids) if st_id == stage_id]
                ds_bounds = ds.bounds
                if len(filtered_idx) > 0:
                    rep_bounds = [ds_bounds] * len(filtered_idx)
                    stage_bounds.extend(rep_bounds)
                    if ds_bounds is not None and all_none is True:
                        all_none = False
            ds_stage_bounds.append(stage_bounds)
        if all_none is True:
            results = None
        else:
            results = ds_stage_bounds
        return results

    def get_full_bounds(self) -> FullBouds:
        """
        Flatten the stage-wise bounds structure into a single list.

        Returns:
            FullBouds: Flattened list containing the bounds associated
                with all variables across all stages.

                Returns None if no bounds are defined.
        """
        stage_bounds = self.get_stage_bounds()
        if stage_bounds is None:
            results = None
        else:
            results = [x for st in stage_bounds for x in st]
        return results

    def create_stages_columns_mapping(self) -> DatasetMappings:
        """
        Create the mapping between datasets, stages, and column positions.

        This method constructs a mapping describing how the columns of the
        full stage-ordered matrix are associated with each dataset and stage.
        Column indices are assigned according to the ordering produced by
        `get_stage_values`.

        Returns:
            DatasetMappings:
                List of dataset mappings. Each mapping contains:
                    - dataset: dataset name
                    - columns: column indices associated with the dataset
                    - stage_ids: stages in which the dataset appears
        """
        results: Dict[str, Dict[str, List[int]]] = {}
        unique_stage_ids = self.get_sorted_stage_ids()
        i_col = 0
        for stage_id in unique_stage_ids:
            for ds in self.datasets:
                num_cols = len([st for st in ds.stage_ids if stage_id == st])
                if num_cols > 0:
                    ini = i_col
                    end = i_col + num_cols
                    if ds.name in results.keys():
                        results[ds.name]["columns"].extend([i for i in range(ini, end)])
                        if stage_id not in results[ds.name]["stage_ids"]:
                            results[ds.name]["stage_ids"].extend([stage_id] * (end - ini))
                    else:
                        results[ds.name] = {
                            "columns": [i for i in range(ini, end)],
                            "stage_ids": [stage_id] * (end - ini),
                        }
                    i_col = end
        stages_columns_mapping: DatasetMappings = []
        for name in results.keys():
            stages_columns_mapping.append(
                {
                    "dataset": name,
                    "columns": results[name]["columns"],
                    "stage_ids": results[name]["stage_ids"],
                }
            )
        return stages_columns_mapping

    def get_num_variables_per_stage(self) -> List[int]:
        """
        Compute the number of variables associated with each stage.

        Returns:
            List[int]: List containing the number of variables for each stage.
                Each position corresponds to a stage in sorted order.
        """
        results = []
        unique_stage_ids = self.get_sorted_stage_ids()
        for stage_id in unique_stage_ids:
            n_var = 0
            for ds in self.datasets:
                n_var += len([st for st in ds.stage_ids if st == stage_id])
            results.append(n_var)
        return results
