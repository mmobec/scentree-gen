import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from scentree.io import MapColumns, MapColsNames
from typing import Dict, List, Optional, Self


class DatasetSpec(BaseModel):
    """
    Container for the minimal information required to describe a dataset.

    Attributes:
        values (NDArray[np.float64]): The matrix containing the data.
        stages (List[int]): The stage each column belongs to.
        priority (List[int]): The priority of each column within its stage.
            It is used to break ties.
        all_positive (bool): Whether all columns are required to be positive.
            Default is False.
    """

    values: NDArray[np.float64]
    stages: List[int]
    priority: List[int]
    all_positive: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """
        Validate the internal consistency of the dataset specification.

        This validator ensures that the metadata describing the dataset
        (`stages` and `priority`) is consistent with the structure of
        `values`.

        The following conditions must hold:
            - `stages` and `priority` must have the same length.
            - The length of `stages` must match the number of columns in `values`.
            - The length of `priority` must match the number of columns in `values`.
            - The prority must be unique within each stage.

        Returns:
            Self: The validated instance.

        Raises:
            ValueError: If any of the consistency conditions are violated.
        """
        if len(self.stages) != len(self.priority):
            raise ValueError("`stages` and `priority` must be of the same length")
        if len(self.stages) != self.values.shape[1]:
            raise ValueError(
                "The length of `stages` must be equal to the number of columns in `values`"
            )
        if len(self.priority) != self.values.shape[1]:
            raise ValueError(
                "The length of `priority` must be equal to the number of columns in `values`"
            )
        # Raise error if there is the same priority for the same stage
        stage_priority = set()
        i = 0
        total = len(self.priority)
        error_idx = None
        while i < total and error_idx is None:
            stage = self.stages[i]
            priority = self.priority[i]
            if (stage, priority) in stage_priority:
                error_idx = (stage, priority)
            else:
                stage_priority.add((stage, priority))
            i += 1
        if error_idx is not None:
            raise ValueError(f"Priority {error_idx[1]} is repeated in stage {error_idx[0]}")
        return self


class DatasetLoader(BaseModel):
    """
    Loader for the information required to represent a multi-stage stochastic programming problem.

    Attributes:
        datasets_information (Dict[str, DatasetSpec]): Dictionary with all datasets
            involved in the problem.
        stage_ids (List[int]): Identifier of each stage. It is automatically populated
            from `datasets_information`.
        stage_data (List[NDArray[np.float64]]): Data matrix corresponding to each stage.
            It is automatically populated from `datasets_information`.
        dataset_names (List[List[str]]): The dataset names each column belongs to.
            It is automatically populated from ` datasets_information`.
    """

    datasets_information: Dict[str, DatasetSpec]
    stage_ids: List[int] = Field(default_factory=list)
    stage_data: List[NDArray[np.float64]] = Field(default_factory=list)
    dataset_names: List[List[str]] = Field(default_factory=list)

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
        n = None
        reference_key = None
        keys = self.datasets_information.keys()
        for k in keys:
            values = self.datasets_information[k].values
            if n is None:
                n = values.shape[0]
                reference_key = k
            else:
                if n != values.shape[0]:
                    raise ValueError(
                        f"""Dataset in {reference_key} contains {n} rows.
                        Dataset in {k} contains {values.shape[0]} rows.
                        All datasets must contain the same number of rows"""
                    )
        return self

    @model_validator(mode="after")
    def validate_priority(self) -> Self:
        """
        Validate that priorities are unique within each stage.

        This validator checks that the combination of `stage` and `priority`
        is unique across all datasets in `datasets_information`. Each stage
        must not contain duplicated priority values, ensuring a consistent
        ordering within stages.

        Raises:
            ValueError: If a priority value is repeated within the same stage
                across any dataset.

        Returns:
            Self: The validated model instance.
        """
        keys = list(self.datasets_information.keys())
        total_keys = len(self.datasets_information.keys())
        i = 0
        error_idx = None
        stage_priority = set()
        while i < total_keys and error_idx is None:
            key = keys[i]
            priorities = self.datasets_information[key].priority
            stages = self.datasets_information[key].stages
            for stage, priority in zip(stages, priorities):
                if (stage, priority) in stage_priority:
                    error_idx = (stage, priority)
                else:
                    stage_priority.add((stage, priority))
            i += 1
        if error_idx is not None:
            raise ValueError(f"Priority {error_idx[1]} is repeated in stage {error_idx[0]}")
        return self

    @model_validator(mode="after")
    def fill_stage_ids(self) -> Self:
        """
        Populate the list of stage identifiers (`stage_ids`) from all datasets.

        This method iterates over all datasets in `datasets_information` and
        collects every unique stage present in the datasets. The resulting list
        of stage identifiers is sorted in ascending order.

        Notes:
            - This is not a validation in the strict sense; it automatically
            fills an attribute based on existing data.
            - Duplicate stages are ignored.

        Returns:
            Self: The model instance with the `stage_ids` attribute populated.
        """
        all_keys = self.datasets_information.keys()
        for key in all_keys:
            stages = self.datasets_information[key].stages
            for stage in stages:
                if stage not in self.stage_ids:
                    self.stage_ids.append(stage)
        self.stage_ids.sort()
        return self

    @model_validator(mode="after")
    def fill_stage_data(self) -> Self:
        """
        Populate the `stage_data` attribute with matrices corresponding to each stage.

        This method iterates over all stage identifiers in `stage_ids` and collects
        the columns from each dataset in `datasets_information` that belong to the
        current stage. Columns are sorted according to their `priority` within the
        stage. The resulting matrices for each stage are stored in `stage_data`,
        and the corresponding ordered dataset names are stored in `dataset_names`.

        Returns:
            Self: The model instance with the `stage_data` attribute populated.
        """
        variables_sorted = []
        dataset_names: List[List[str]] = []
        for stage in self.stage_ids:
            variables_stage = []
            priorities_stage = []
            dataset_names_stage = []
            for key in self.datasets_information.keys():
                info = self.datasets_information[key]
                idx_variables_stage = [i for i, value in enumerate(info.stages) if value == stage]
                dataset_name = len(idx_variables_stage) * [key]
                # In case there is data for this stage, we sort the data accordingly
                if len(idx_variables_stage) > 0:
                    priority_variables = [info.priority[i] for i in idx_variables_stage]
                    priorities_stage.extend(priority_variables)
                    variables = info.values[:, idx_variables_stage]
                    variables_stage.append(variables)
                    dataset_names_stage.extend(dataset_name)
            variables_stage_matrix = np.concatenate(variables_stage, axis=1)
            idx_priorities_sorted = np.argsort(priorities_stage)
            dataset_names_stage_sorted = [dataset_names_stage[i] for i in idx_priorities_sorted]
            dataset_names.append(dataset_names_stage_sorted)
            variables_sorted.append(variables_stage_matrix[:, idx_priorities_sorted])
        self.stage_data = variables_sorted
        self.dataset_names = dataset_names
        return self

    def to_stage_wise_loader(self) -> "StageWiseLoader":
        """
        Convert the dataset loader into a stage-wise loader.

        This method creates a `StageWiseLoader` instance where each element
        of `data` corresponds to the matrix of variables for a stage
        (from `stage_data`), `stage_names` contains the names of the
        datasets associated with each stage and `mapping_columns_stages` maps
        each column to its stage.

        Returns:
            StageWiseLoader: A new instance of `StageWiseLoader` containing
                the stage-wise data and stage names.
        """
        stage_loader = StageWiseLoader(
            data=self.stage_data,
            stage_ids=self.stage_ids,
            mapping_columns_stages=self.dataset_names,
        )
        return stage_loader


class StageWiseLoader(BaseModel):
    """
    Represents a multi-stage dataset organized by stages.

    Attributes:
        data (List[NDArray[np.float64]]): List of matrices, one per stage, where
            each matrix contains the variables for that stage. All matrices
            must have the same number of rows.
        stage_ids (List[int]): List of identifiers for each stage. Must have
            the same length as `data`.
        mapping_columns_stages (Optional[List[List[str]]]): Optional mapping
            of dataset names for each column in each stage. Each inner list
            must have the same length as the number of columns of the
            corresponding stage matrix.
        num_variables_per_stage (List[int]): Number of variables (columns)
            for each stage. It is automatically populated from `data`.
    """

    data: List[NDArray[np.float64]]
    stage_ids: List[int]
    mapping_columns_stages: Optional[List[List[str]]] = None
    num_variables_per_stage: List[int] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_stage_ids(self) -> Self:
        """
        Validate that `stage_ids` and `data` have the same length.

        Raises:
            ValueError: If the number of stage identifiers does not match
                the number of stage matrices.

        Returns:
            Self: The validated model instance.
        """
        if len(self.stage_ids) != len(self.data):
            raise ValueError("`stage_ids` and `data` must be of the same length.")
        return self

    @model_validator(mode="after")
    def validate_mapping_columns_stages(self) -> Self:
        """
        Validate that `mapping_columns_stages` matches the structure of `data`.

        If provided, the number of elements in `mapping_columns_stages` must
        match the number of stage matrices.

        Raises:
            ValueError: If the length of `mapping_columns_stages` does not
                match the length of `data`.

        Returns:
            Self: The validated model instance.
        """
        if self.mapping_columns_stages is not None:
            if len(self.mapping_columns_stages) != len(self.data):
                raise ValueError("`mapping_columns_stages` and `data` must be of the same length.")
        return self

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        """
        Validate the integrity of the stage data.

        This validator checks that:
        - All stage matrices have the same number of rows.
        - No matrix contains NaN values.
        - If `mapping_columns_stages` is provided, each mapping list has
          the same length as the number of columns of its corresponding matrix.

        Raises:
            ValueError: If any stage matrix has a different number of rows.
            ValueError: If any matrix contains NaN values.
            ValueError: If the column mapping length does not match the number
                of columns in the corresponding matrix.

        Returns:
            Self: The validated model instance.
        """
        position = 0
        n = self.data[position].shape[0]
        for i, value in enumerate(self.data):
            if value.shape[0] != n:
                raise ValueError(
                    f"""Dataset in position {position} contains {n} rows.
                    Dataset in position {i} contains {value.shape[0]}.
                    All datasets must contain the same number of rows."""
                )
            if np.isnan(value).any():
                raise ValueError(f"Dataset in position {i} contains NAN values.")
            if self.mapping_columns_stages is not None:
                if len(self.mapping_columns_stages[i]) != value.shape[1]:
                    raise ValueError(
                        f"""Dataset in position {i} contains {value.shape[1]} columns.
                        List of `mapping_columns_stages` in position {i} contains
                        {len(self.mapping_columns_stages[i])} values. Both numbers must be equal."""
                    )
        return self

    @model_validator(mode="after")
    def fill_attributes(self) -> Self:
        """
        Populate derived attributes.

        This method computes the number of variables (columns) for each stage
        and stores the result in `num_variables_per_stage`.

        Returns:
            Self: The model instance with derived attributes populated.
        """
        self.num_variables_per_stage = [X.shape[1] for X in self.data]
        return self

    def create_full_data(self) -> NDArray[np.float64]:
        """
        Concatenate all stage matrices into a single matrix along columns.

        Returns:
            NDArray[np.float64]: A matrix containing all variables from all stages,
                concatenated column-wise.
        """
        return np.concatenate(self.data, axis=1)

    def build_stages_columns_mapping(self) -> MapColsNames:
        """
        Build a mapping between dataset names and their column indices.

        This method aggregates the information in `mapping_columns_stages`
        and generates a list of mappings. Each mapping contains a dataset
        name and the list of column indices (in the concatenated matrix)
        that belong to that dataset.

        If `mapping_columns_stages` is not provided, the method returns None.

        Returns:
            MapColsNames: A list of mappings where each element contains:
                - dataset: Name of the dataset.
                - columns: List of column indices associated with that dataset.
            None: If `mapping_columns_stages` is not defined.
        """
        results = None
        if self.mapping_columns_stages is not None:
            i = 0
            results = []
            dict_mapping: Dict[str, List[int]] = {}
            for stage_names in self.mapping_columns_stages:
                for name in stage_names:
                    if name in dict_mapping.keys():
                        dict_mapping[name].append(i)
                    else:
                        dict_mapping[name] = [i]
                    i += 1
            for name in dict_mapping.keys():
                res_dict: MapColumns = {"dataset": name, "columns": dict_mapping[name]}
                results.append(res_dict)
        return results
