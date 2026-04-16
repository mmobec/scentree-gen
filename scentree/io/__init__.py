from typing import List, Optional, TypedDict


class MapColumns(TypedDict):
    """
    Mapping between a dataset and the ordering of its columns.

    Attributes:
        dataset (str): Name of the dataset.
        columns (List[int]): List of column indices defining their order.
    """

    dataset: str
    columns: List[int]


# Optional list of dataset-to-column mappings
MapColsNames = Optional[List[MapColumns]]
