Note that some of the classes and methods used in this example support additional input parameters that can modify their behavior. See the [API documentation](../api/fan_generator.md) for more details.

Scenario fan generation is decoupled from tree construction to provide greater flexibility. Users may already have their own procedure for generating scenario fans, in which case they are not required to use `scentree` for this step. Instead, they can generate the scenario fans externally and use this package solely to construct the corresponding scenario tree.

## Load the data
This example illustrates how to use scentree to generate four scenario trees. The stochastic problem consists of three stages and involves two datasets, represented by `X_a` and `X_b`. The example uses simulated data. If the data is stored in multiple files, they must first be loaded and converted to `NumPy` arrays.

Replace this step with the appropriate code to load the data from the files. The package expects the data to be arranged row-wise, with each observation stored in a separate row. In this example, each row represents a day.

The following table shows the relationship between the columns of each dataset and the stages.

| Dataset   | # Columns | Stage 1 columns | Stage 2 columns | Stage 3 columns |
|-----------|-----------|-----------------|-----------------|-----------------|
| `X_a`     | 5         | 0, 1, 2         | 3               | 4               |
| `X_b`     | 3         | 0               | 1               | 2               |


```python
import numpy as np
np.random.seed(42)
X_a = np.random.normal(size=(50, 5))
X_b = np.random.normal(size=(50, 3))
```

The package includes a data loader that converts `NumPy` arrays into an appropriate internal representation.
```python
from scentree.io.loader import Dataset, DatasetsLoader
datasets = [
    Dataset(
        name="a",
        values=X_a,
        stage_ids=[1, 1, 1, 2, 3],
    ),
    Dataset(
        name="b",
        values=X_b,
        stage_ids=[1, 2, 3],
    )
]
dataset_loader = DatasetsLoader(datasets=datasets)
full_values = dataset_loader.get_full_values()
num_variables_per_stage = dataset_loader.get_num_variables_per_stage()
stage_ids = dataset_loader.get_sorted_stage_ids()
map_columns_names = dataset_loader.create_stages_columns_mapping()
```
The variable `full_values` is a matrix with 50 rows and 8 columns. It is obtained by concatenating `X_a` and `X_b` column-wise. `num_variables_per_stage` is a list containing the number of random variables in each stage, while `stage_ids` is a list of unique stage identifiers. Finally, `map_columns_names` is used to filter `X_a` and `X_b` from `full_values`. For more details, see the [loader API documentation](../api/loader.md).

## Scenario fan generation
The `StageManager` class manages scenario fan generation by selecting the best predictive model. In this example, a total of four scenario fans are generated, each containing 10 scenarios.
```python
from scentree.fan_generator import StageManager
num_fans = 4
num_scenarios = 10
stage_manager = StageManager()
scenario_fans = stage_manager.generate_scenario_fans(
    X=full_values,
    num_fans=num_fans,
    num_scenarios=num_scenarios,
)
```

Note that the output is a dictionary. Inspect the returned object to access the generated scenarios. In this example, the `scenario_fans` variable is a list of length four, where each element contains a set of scenarios.

This step involves randomly sampling the estimated residuals. To ensure reproducibility, the `generate_scenario_fans` method accepts an optional `seed` parameter. Using the same seed, along with identical input data and model configuration, will generate the same scenario fans. For more details, see the [scenario fan generation API documentation](../api/fan_generator.md).

## Scenario tree construction
The current version of the package implements only the Forward Tree Construction algorithm (FTC). Additional scenario tree construction algorithms will be incorporated in future releases.
```python
from scentree.tree_construction.ftc import FTC
tree_builder = FTC(
    scenarios=scenario_fans["scenarios"],
    num_variables_per_stage=num_variables_per_stage,
    stage_ids=stage_ids
)
scenario_trees = tree_builder.generate_scenario_trees(initial_stage_id_to_cluster=1)
```

The output is a list of length four, where each element is a dictionary containing the constructed scenario tree together with additional information produced during the scenario tree construction process. See the [tree construction API documentation](../api/tree_construction.md) for a detailed description of the returned data structure and its contents.

## Store the output
Finally, the output is stored in a JSON file. The user can choose whether to store all scenario trees in a single JSON file or in multiple files. For this purpose, a dedicated output folder is created. For more details, see the [output API documentation](../api/output.md).

If the optimization problem to be solved concerns an energy community participating in electricity markets, the resulting output can be directly used with the open source [`secoem`](https://github.com/mmobec/secoem) package.
```python
from scentree.io.writer import save_json
save_json(
    output_dir=".",
    num_stages=len(stage_ids),
    predicted_value=scenario_fans["predicted_values"],
    observed_value=scenario_fans["observed_values"],
    scenario_trees=scenario_trees,
    mapping_datasets_columns=map_columns_names,
    multiple_files=True,
)
```
