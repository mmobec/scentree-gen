from typing import List, TypedDict


class DatasetMapping(TypedDict):
    """
    Mapping between a dataset and the ordering of its columns.

    Attributes:
        dataset (str): Name of the dataset.
        columns (List[int]): List of column indices defining their order.
        stages (List[int]): List of column stages.
    """

    dataset: str
    columns: List[int]
    stage_ids: List[int]


DatasetMappings = List[DatasetMapping]
