import numpy as np
from pydantic import BaseModel, Field, model_validator
from typing import Dict, List, Self, Optional


class DatasetSpec(BaseModel):
    """
    Container for the minimal information required to describe a dataset.

    Attributes:
        values (np.ndarray): The matrix containing the data.
        stages (List[int]): The stage each column belongs to.
        priority (List[int]): The priority of each column within its stage.
            It is used to break ties.
        all_positive (bool): Whether all columns are required to be positive.
            Default is False.
    """

    values: np.ndarray
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
        return self


class DatasetLoader(BaseModel):
    """
    Loader for the information required to represent a multi-stage stochastic programming problem.

    Attributes:
        datasets_information (Dict[str, DatasetSpec]): Dictionary with all datasets
            involved in the problem.
        stage_ids (List[int]): Identifier of each stage. It is automatically populated
            from`datasets_information`.
        stage_data (List[np.ndarray]): Data matrix corresponding to each stage.
            It is automatically populated from`datasets_information`.
    """

    datasets_information: Dict[str, DatasetSpec]
    stage_ids: List[int] = Field(default_factory=list)
    stage_data: List[np.ndarray] = Field(default_factory=list)

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
        stage. The resulting matrices for each stage are stored in `stage_data`.

        Returns:
            Self: The model instance with the `stage_data` attribute populated.
        """
        variables_sorted = []
        for stage in self.stage_ids:
            variables_stage = []
            priorities_stage = []
            for key in self.datasets_information.keys():
                info = self.datasets_information[key]
                idx_variables_stage = [i for i, value in enumerate(info.stages) if value == stage]
                # In case there is data for this stage, we sort the data accordingly
                if len(idx_variables_stage) > 0:
                    priority_variables = [info.priority[i] for i in idx_variables_stage]
                    priorities_stage.extend(priority_variables)
                    variables = info.values[:, idx_variables_stage]
                    variables_stage.append(variables)
            variables_stage = np.concatenate(variables_stage, axis=1)
            idx_priorities_sorted = np.argsort(priorities_stage)
            variables_sorted.append(variables_stage[:, idx_priorities_sorted])
        self.stage_data = variables_sorted
        return self

    def to_stage_wise_loader(self) -> "StageWiseLoader":
        """
        Convert the dataset loader into a stage-wise loader.

        This method creates a `StageWiseLoader` instance where each element
        of `data` corresponds to the matrix of variables for a stage
        (from `stage_data`), and `stage_names` contains the names of the
        datasets associated with each stage.

        Returns:
            StageWiseLoader: A new instance of `StageWiseLoader` containing
                the stage-wise data and stage names.
        """
        stage_names = [name for name in self.datasets_information.keys()]

        stage_loader = StageWiseLoader(
            data=self.stage_data, stage_ids=self.stage_ids, stage_names=stage_names
        )
        return stage_loader


class StageWiseLoader(BaseModel):
    """
    Represents a multi-stage dataset organized by stages.

    Each stage contains a matrix of variables, and optionally each stage
    can have a name. This class ensures that all stage matrices have
    the same number of rows and allows concatenating all stages into
    a single matrix.

    Attributes:
        data (List[np.ndarray]): List of matrices, one per stage, where
            each matrix contains the variables for that stage. All matrices
            must have the same number of rows.
        stage_names (Optional[List[str]]): Optional list of names for each
            stage. Must have the same length as `data` if provided.
    """

    data: List[np.ndarray]
    stage_ids: List[int]
    stage_names: Optional[List[str]] = None
    num_variables_per_stage: List[int] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_data(self) -> Self:
        """
        Validate that all stage matrices have the same number of rows.

        Raises:
            ValueError: If any stage matrix has a different number of rows
                than the first stage.

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
        return self

    @model_validator(mode="after")
    def validate_stage_ids(self) -> Self:
        if len(self.stage_ids) != len(self.data):
            raise ValueError("`data` and `stage_ids` must be of the same length")
        return self

    @model_validator(mode="after")
    def validate_stage_names(self) -> Self:
        """
        Validate that `stage_names` has the same length as `data`.

        Raises:
            ValueError: If `stage_names` is provided and its length does
                not match the number of stages in `data`.

        Returns:
            Self: The validated model instance.
        """
        if self.stage_names is not None and len(self.stage_names) != len(self.data):
            raise ValueError("`data` and `stage_names` must be of the same length")
        return self

    @model_validator(mode="after")
    def fill_attributes(self) -> Self:
        self.num_variables_per_stage = [X.shape[1] for X in self.data]
        return self

    def create_full_data(self) -> np.ndarray:
        """
        Concatenate all stage matrices into a single matrix along columns.

        Returns:
            np.ndarray: A matrix containing all variables from all stages,
                concatenated column-wise.
        """
        return np.concatenate(self.data, axis=1)
