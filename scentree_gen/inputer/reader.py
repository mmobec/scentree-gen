import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List


class Reader(BaseModel):
    data: List[np.ndarray]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data", mode="before")
    @classmethod
    def check_is_list_2d_ndarray(cls, v):
        if not isinstance(v, list):
            raise TypeError("A List is expected in data")
        for i, element in enumerate(v):
            if not isinstance(element, np.ndarray):
                raise TypeError(f"Element at index {i} is not a numpy.ndarray.")
            if element.ndim != 2:
                raise ValueError(f"Element at index {i} does not have 2 dimensions")
        return v
