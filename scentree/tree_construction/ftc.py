import numpy as np
from copy import copy
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from scentree.tree_construction import Node, Tree, TreeInfoMap
from scipy.spatial import distance_matrix
from typing import Dict, List, Optional, Self, Tuple, TypedDict


class Graph(TypedDict):
    """Graph structure representing the scenario tree.

    This dictionary stores the relationship between nodes (identified by
    integer IDs) and their corresponding `(stage, representative)` pairs,
    as well as the edges defining the tree structure.

    Attributes:
        ids (Dict[int, Tuple[Optional[int], Optional[int]]]):
            Mapping from node IDs to `(stage, representative)` pairs.
            The root node is typically represented as `(-1, (None, None))`.

        edges (List[Tuple[int, int]]):
            List of directed edges `(parent_id, child_id)` defining the
            tree structure.
    """

    ids: Dict[int, Tuple[Optional[int], Optional[int]]]
    edges: List[Tuple[int, int]]


class FTC(BaseModel):
    """
    Obtain clusters accoring to the Forward Tree Construction algorithm (FTC).

    Attributes:
        scenarios (List[NDArray[np.float64]]): List of scenarios.
        stage_ids (List[int]): List of stages IDs of the stochastic problem.
        num_variables_per_stage (List[int]): Number of random variables per each stage

        Computed Attributes:
            num_trees (Optional[int]): Number of generated trees.
            num_scenarios (Optional[int]): Total number of scenarios.
            scenario_ids (List[int]): Identifiers of the scenarios.
    """

    scenarios: List[NDArray[np.float64]]
    stage_ids: List[int]
    num_variables_per_stage: List[int]
    num_trees: Optional[int] = None
    num_scenarios: Optional[int] = None
    scenario_ids: List[int] = Field(default_factory=list)
    _scenario_index_map: Dict[int, int] = PrivateAttr(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_scenarios(self) -> Self:
        """Validate that all scenarios have the same number of rows.

        Raises:
            ValueError: If the number of rows in any scenario does not match the first scenario.

        Returns:
            Self: The model instance (`self`) if validation passes.
        """
        idx_reference = 0
        n_reference = self.scenarios[idx_reference].shape[0]
        for i in range(len(self.scenarios)):
            if self.scenarios[i].shape[0] != n_reference:
                raise ValueError(
                    f"""The number of rows of scenarios in position {idx_reference} does not
                    match the number of rows of scenarios in position {i}"""
                )
        return self

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Ensure that `stage_ids` and `num_variables_per_stage` have the same length.

        Raises:
            ValueError: If `stage_ids` and `num_variables_per_stage` do not have the same length.

        Returns:
            Self: The model instance (`self`) if validation passes.
        """
        if len(self.stage_ids) != len(self.num_variables_per_stage):
            raise ValueError("`stage_ids` and `num_variables_per_stage` must have the same length")
        return self

    @model_validator(mode="after")
    def fill_attributes(self) -> Self:
        """Automatically fill computed attributes after model initialization.

        It sets the following attributes based on the input `scenarios`:

        - `num_trees`: Number of trees (equal to the number of scenarios).
        - `num_scenarios`: Number of rows in each scenario.
        - `scenario_ids`: Sequential identifiers for each scenario (starting from 1).

        Returns:
            Self: The model instance (`self`) with updated attributes.
        """
        self.num_trees = len(self.scenarios)
        self.num_scenarios = self.scenarios[0].shape[0]
        self.scenario_ids = list(range(1, self.scenarios[0].shape[0] + 1))
        self._scenario_index_map = {v: i for i, v in enumerate(self.scenario_ids)}
        return self

    @staticmethod
    def compute_weighted_norm(
        x: NDArray[np.float64], weights: NDArray[np.float64], r: float, compute_r_root: bool = True
    ) -> float:
        """Compute weighted norm.

        Args:
            x (NDArray[np.float64]): vector containing the data.
            weights (NDArray[np.float64]): vector containing the weights.
            r (float): the exponent.
            compute_r_root (bool, optional): if the r-root is computed at the end. Default to True.

        Returns:
            float: The weighted norm of `x` according to `weight` and exponent `r`.
        """
        x_abs = np.abs(x)
        x_pow = np.power(x_abs, r)
        prod = np.multiply(x_pow, weights)
        result = np.sum(prod)
        if compute_r_root:
            result = np.power(result, 1 / r)
        return float(result)

    def mapping_stages_columns(self) -> Dict[int, Tuple[int, int]]:
        """Return the column indices for each stage.

        Each stage is mapped to a tuple containing the starting and ending
        column indices in the overall variable array.

        Returns:
            Dict[int, Tuple[int, int]]: Mapping from `stage_id` to `(start_column, end_column)`.
        """
        mapping = dict()
        end_stage = 0
        min_stage_id = min(self.stage_ids)
        # An initial stage is create with variable of length 0. This represents the root
        stage_ids = [x for x in self.stage_ids]
        stage_ids.insert(0, min_stage_id - 1)
        num_variables_per_stage = [x for x in self.num_variables_per_stage]
        num_variables_per_stage.insert(0, 0)
        for stage_id, num_variables in zip(stage_ids, num_variables_per_stage):
            ini_stage = end_stage
            end_stage = ini_stage + num_variables
            mapping[stage_id] = (ini_stage, end_stage)
        return mapping

    def compute_stages_thresholds(
        self,
        scenarios: NDArray[np.float64],
        map_stages_columns: Dict[int, Tuple[int, int]],
        full_stage_ids: List[int],
        probability_scenarios: NDArray[np.float64],
        r: float,
        initial_stage_id_to_cluster: Optional[int],
    ) -> Dict[int, float]:
        """Compute stage-specific clustering thresholds for scenarios.

        For each stage, a threshold is computed as the weighted distance between
        the closest scenario and all other scenarios in that stage. The first
        step identifies the closest scenario based on `initial_stage_id_to_cluster`,
        which is then used to calculate distances to the remaining scenarios.

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.
            full_stage_ids (List[int]): List of stages of the stochastic problem.
            probability_scenarios (NDArray[np.float64]): Probability of each scenario.
            r (float): Exponent used in the weighted norm computation.
            initial_stage_id_to_cluster (Optional[int]): Stage ID from which clustering starts.

        Returns:
            Dict[int, float]: Mapping from stage ID to threshold value for clustering.
        """

        dist_rel = 0.8
        dist_increment = 0.0
        threshold_stages = {}
        # Compute the distance matrix taking into account the first stage to cluster
        if initial_stage_id_to_cluster is not None:
            # Haig de buscar l'index dins l'attribuit stages_id
            idx_column_initial_cluster = map_stages_columns[initial_stage_id_to_cluster][0]
            data_matrix = scenarios[:, idx_column_initial_cluster:]
            idx_stage_id_initial_cluster = full_stage_ids.index(initial_stage_id_to_cluster)
            stage_ids_to_cluster = full_stage_ids[idx_stage_id_initial_cluster:]
            distance_global = distance_matrix(data_matrix, data_matrix, p=1)
            # Compute the weighted norm
            weighted_norm = np.apply_along_axis(
                func1d=self.compute_weighted_norm,
                axis=0,
                arr=distance_global,
                weights=probability_scenarios,
                r=r,
            )
            # Get the closest scenario
            idx_closest_scenario = np.argmin(weighted_norm)
            # Get the distance between all scenarios and the closest one for each stage
            for stage in stage_ids_to_cluster:
                stage_mapping = map_stages_columns[stage]
                ini, end = stage_mapping
                data_stage = scenarios[:, ini:end]
                data_closest = np.reshape(scenarios[idx_closest_scenario, ini:end], (1, -1))
                distance_stage_closest = np.ravel(distance_matrix(data_closest, data_stage, p=1))
                wns = self.compute_weighted_norm(
                    x=distance_stage_closest, weights=probability_scenarios, r=r
                )
                current_difference = dist_rel - dist_increment
                threshold_stages[stage] = wns * current_difference
                dist_increment += 0.005
        return threshold_stages

    def update_non_clustering_stages(
        self,
        scenarios: NDArray[np.float64],
        map_stages_columns: Dict[int, Tuple[int, int]],
        full_stage_ids: List[int],
        prob_scenarios_stages: NDArray[np.float64],
        initial_stage_id_to_cluster: Optional[int],
        tree: TreeInfoMap,
        representatives: Dict[int, List[int]],
        Scen0: NDArray[np.float64],
        graph: Graph,
    ) -> None:
        """Populate data structures for stages that are not clustered.

        This method updates the tree, representatives, scenario matrix (`Scen0`),
        probability matrix, and graph structure for stages that are not subject
        to clustering (i.e., stages before `initial_stage_id_to_cluster`).

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.
            full_stage_ids (List[int]): List of stages of the stochastic problem.
            prob_scenarios_stages (NDArray[np.float64]): Matrix containing the probability of each
                scenario at each stage.
            initial_stage_id_to_cluster (Optional[int]): Stage ID from which clustering starts.
            tree (TreeInfoMap): Dictionary storing clusters for each stage.
            representatives (Dict[int, List[int]]): Representative scenarios for each stage.
            Scen0 (NDArray[np.float64]): The resulting data from the cluster process, i.e, the tree.
            graph (Graph): The graph representing the tree.
        """
        idx_representative = int(len(self.scenario_ids) / 2) - 1
        scenario_representative = self.scenario_ids[idx_representative]
        if initial_stage_id_to_cluster is not None:
            idx_stage_initial_cluster = full_stage_ids.index(initial_stage_id_to_cluster)
            non_clustering_stages = full_stage_ids[:idx_stage_initial_cluster]
        else:
            idx_stage_initial_cluster = None
            non_clustering_stages = full_stage_ids[:]
        predecessor_id = -1
        current_id = 0
        graph["ids"][predecessor_id] = (None, None)
        for i_stage, stage in enumerate(non_clustering_stages):
            key = (stage, scenario_representative)
            tree[key] = self.scenario_ids[:]
            representatives[stage] = [scenario_representative]
            stage_mapping = map_stages_columns[stage]
            ini = stage_mapping[0]
            end = stage_mapping[1]
            Scen0[:, ini:end] = scenarios[:, ini:end]
            prob_scenarios_stages[idx_representative, i_stage] = len(tree[key]) / len(
                self.scenario_ids
            )
            graph["ids"][current_id] = key
            graph["edges"].append((predecessor_id, current_id))
            predecessor_id = current_id
            current_id += 1
        return None

    def compute_distance_stage(
        self,
        scenarios: NDArray[np.float64],
        stage_id: int,
        map_stages_columns: Dict[int, Tuple[int, int]],
    ) -> NDArray[np.float64]:
        """Compute the distance matrix for a given stage.

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            stage_id (int): The stage ID.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.

        Returns:
            NDArray[np.float64]: The pairwise distances matrix between scenarios for the
                given stage.
        """
        stage_mapping = map_stages_columns[stage_id]
        ini, end = stage_mapping
        data_filtered = scenarios[:, ini:end]
        dist_matrix: NDArray[np.float64] = distance_matrix(data_filtered, data_filtered, p=1)
        return dist_matrix

    def get_vertex_id(self, graph: Graph, stage_id: int, representative_id: int) -> int:
        """Get the identifier of a vertex given the stage and the representative.

        Args:
            graph (Graph): The graph representing the tree.
            stage_id (int): the stage.
            representative_id (int): Representative scenario ID associated with the vertex.

        Raises:
            ValueError: If no matching vertex is found.

        Returns:
            int: The identifier of the vertex.
        """
        element_to_find = (stage_id, representative_id)
        for vertex_id, value in graph["ids"].items():
            if value == element_to_find:
                return vertex_id
        raise ValueError(f"Vertex ({stage_id}, {representative_id}) not found in graph.")

    def get_representative(
        self,
        distance: NDArray[np.float64],
        weights: NDArray[np.float64],
        selected_scenario_ids: List[int],
        r: float,
    ) -> int:
        """Return the representative scenario for a given step in a stage.

        Args:
            distance (NDArray[np.float64]): The pairwise distances matrix between scenarios.
            weights (NDArray[np.float64]): vector containing the weight of each scenario.
            selected_scenario_ids (List[int]): List of scenario IDs considered
                for representative selection.
            r (float): Exponent used in the weighted norm computation.

        Returns:
            int: The chosen representative.
        """
        idx_selected_scenarios = [self._scenario_index_map[x] for x in selected_scenario_ids]
        # For each scenario, compute the weghted norm of the distance
        idx_selected_scenarios = [self.scenario_ids.index(x) for x in selected_scenario_ids]
        distance_filtered = distance[idx_selected_scenarios, :][:, idx_selected_scenarios]
        weights_filtered = weights[idx_selected_scenarios]
        weighted_norms = np.apply_along_axis(
            func1d=self.compute_weighted_norm,
            axis=0,
            arr=distance_filtered,
            weights=weights_filtered,
            r=r,
            compute_r_root=False,
        )
        arg_min = np.argmin(weighted_norms)
        representative = selected_scenario_ids[arg_min]
        return representative

    def map_scenarios_to_representatives(
        self,
        representative_ids: List[int],
        selected_scenario_ids: List[int],
        distance: NDArray[np.float64],
    ) -> Dict[int, int]:
        """Relates scenario ids and representatives.

        Args:
            representative_ids (List[int]): List of representative scenario IDs.
            selected_scenario_ids (List[int]): List of scenario IDs to assign.
            distance (NDArray[np.float64]): The pairwise distances matrix between scenarios.

        Raises:
            ValueError: If there is a mismatch in the shape of objects.

        Returns:
            Dict[int, int]: dictionary containing the relationship.
        """
        idx_scenarios = [self._scenario_index_map[x] for x in selected_scenario_ids]
        idx_representatives = [self._scenario_index_map[x] for x in representative_ids]
        distance_filtered = distance[idx_scenarios, :][:, idx_representatives]
        idx_closest_representative = np.argmin(distance_filtered, axis=1)
        scenario_to_representative = {}
        if len(selected_scenario_ids) != idx_closest_representative.shape[0]:
            raise ValueError("Mismatch between selected scenarios and computed representatives.")
        for sc, i_pre in zip(selected_scenario_ids, idx_closest_representative):
            rep = representative_ids[i_pre]
            scenario_to_representative[sc] = sc if sc in representative_ids else rep
        return scenario_to_representative

    def update_tree(
        self,
        scenarios: NDArray[np.float64],
        Scen0: NDArray[np.float64],
        closest_representative: Dict[int, int],
        selected_scenario_ids: List[int],
        map_stages_columns: Dict[int, Tuple[int, int]],
        stage_id: int,
    ) -> None:
        """Build the tree at the given stage.

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            Scen0 (NDArray[np.float64]): The resulting data from the cluster process, i.e, the tree.
            closest_representative (Dict[int, int]): dictionary containing the relationship
                scenario - representative.
            selected_scenario_ids (List[int]): List of scenario IDs considered
                for representative selection.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.
            stage_id (int): The stage ID.
        """
        stage_mapping = map_stages_columns[stage_id]
        ini, end = stage_mapping
        for sc in selected_scenario_ids:
            representative_scenario = closest_representative[sc]
            idx_representative = self._scenario_index_map[representative_scenario]
            idx_scenario = self._scenario_index_map[sc]
            Scen0[idx_scenario, ini:end] = scenarios[idx_representative, ini:end]
        return None

    def compute_delta_norm_tree(
        self,
        scenarios: NDArray[np.float64],
        map_stages_columns: Dict[int, Tuple[int, int]],
        stage_id: int,
        tree_data: NDArray[np.float64],
        weights: NDArray[np.float64],
        r: float,
        selected_scenario_ids: Optional[List[int]] = None,
    ) -> float:
        """Compute the weighted norm of the tree once it has been clustered at the given stage.

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.
            stage_id (int): The stage ID.
            tree_data (NDArray[np.float64]): Array representing the clustered tree data.
            weights (NDArray[np.float64]): vector containing the weight of each scenario.
            r (float): Exponent used in the weighted norm computation.
            selected_scenario_ids (Optional[List[int]]): Subset of scenarios IDs to consider.

        Returns:
            float: Weighteds norm of the difference between scenarios and tree at a given stage,
        """
        if selected_scenario_ids is None:
            selected_scenario_ids = self.scenario_ids
        idx_scenarios = [self._scenario_index_map[sc] for sc in selected_scenario_ids]
        stage_mapping = map_stages_columns[stage_id]
        ini, end = stage_mapping
        scenarios_stage = scenarios[idx_scenarios, ini:end]
        tree_stage = tree_data[idx_scenarios, ini:end]
        weights_filtered = weights[idx_scenarios]
        diff = np.subtract(scenarios_stage, tree_stage)
        # Compute the norm of the difference for each scenarios
        scenarios_norm = np.sum(np.abs(diff), axis=1)
        tree_norm = self.compute_weighted_norm(x=scenarios_norm, weights=weights_filtered, r=r)
        return tree_norm

    def update_probability(
        self,
        probability_matrix: NDArray[np.float64],
        stage_ids: List[int],
        stage_id: int,
        tree: TreeInfoMap,
        representatives: Dict[int, List[int]],
    ) -> None:
        """Update scenario probabilities at a given stage after clustering.

        Args:
            probability_matrix (NDArray[np.float64]): Probability matrix for all scenarios
                and all stages.
            stage_ids (List[int]): List containing all stage IDs.
            stage_id (int): The stage ID.
            tree (TreeInfoMap): Dictionary storing clusters for each stage.
            representatives (Dict[int, List[int]]): Representative scenario IDs for each stage.
        """
        total_scenarios = len(self.scenario_ids)
        stage_index_map = {v: i for i, v in enumerate(stage_ids)}
        idx_stage = stage_index_map[stage_id]
        representatives_stage = representatives[stage_id]
        for rep in representatives_stage:
            key = (stage_id, rep)
            scenarios_representatives = tree[key]
            total_related_scenarios = len(scenarios_representatives)
            idx_rep = self._scenario_index_map[rep]
            probability_matrix[idx_rep, idx_stage] = total_related_scenarios / total_scenarios
        return None

    def filter_edges_by_vertex(
        self, graph: Graph, stage_id: int, representative_id: int, by_current: bool = True
    ) -> List[Tuple[int, int]]:
        """Return edges connected to a vertex defined by stage and representative.

        Args:
            graph (Graph): The graph representing the tree.
            stage_id (int): The stage ID.
            representative_id (int): The representative ID.
            by_current (bool): If True, match edges where the vertex is the target (child).
                If False, match edges where the vertex is the source (parent).

        Returns:
            List[Tuple[int, int]]: List of edges matching the given vertex.
        """
        vertex_id = self.get_vertex_id(graph, stage_id, representative_id)
        edges = graph["edges"]
        edges_filtered = []
        for current_edge in edges:
            idx = 1 if by_current else 0
            edge_id_element = current_edge[idx]
            if edge_id_element == vertex_id:
                edges_filtered.append(current_edge)
        return edges_filtered

    def update_stage(
        self,
        scenarios: NDArray[np.float64],
        map_stages_columns: Dict[int, Tuple[int, int]],
        stage_ids: List[int],
        stage_id: int,
        representatives: Dict[int, List[int]],
        tree: TreeInfoMap,
        graph: Graph,
        Scen0: NDArray[np.float64],
        prob_scenarios_stages: NDArray[np.float64],
        distance_stage: NDArray[np.float64],
    ) -> None:
        """Update a whole stage so that the norm is lower than the treshold.

        Args:
            scenarios (NDArray[np.float64]): Array containing the data of the scenarios.
            map_stages_columns (Dict[int, Tuple[int, int]]): Mapping between stages and columns.
            stage_ids (List[int]): List containing all stage IDs.
            stage_id (int): the stage ID.
            representatives (Dict[int, List[int]]): Representative scenario IDs for each stage.
            tree (TreeInfoMap): Dictionary storing clusters for each stage.
            graph (Graph): The graph representing the tree.
            Scen0 (NDArray[np.float64]): The resulting data from the cluster process, i.e, the tree.
            prob_scenarios_stages (NDArray[np.float64]): Matrix containing the probability of each
                scenario at each stage.
            distance_stage: (NDArray[np.float64]): Matrix distance for the given stage_id.
        """
        # Select a cluster with more than 1 scenario
        clusters = representatives[stage_id]
        clusters_permutated = np.random.permutation(clusters)
        found = False
        i = 0
        total_clusters = clusters_permutated.shape[0]
        scenarios_rep = None
        total_scenarios = None
        rep = None
        while i < total_clusters and not found:
            rep = clusters_permutated[i]
            key = (stage_id, rep)
            scenarios_rep = copy(tree[key])
            total_scenarios = len(scenarios_rep)
            found = total_scenarios > 1
            i += 1
        # Split the cluster by means of selecting a new representative.
        if scenarios_rep is None:
            raise ValueError("Invalid value for `scenarios_rep`")
        new_rep = scenarios_rep[0]
        i = 0
        if total_scenarios is None:
            raise ValueError("`total_scenarios` must be an integer")
        if rep is None:
            raise ValueError("`rep` must be an integer")
        while i < total_scenarios and new_rep == rep:
            new_rep = scenarios_rep[i]
            i += 1
        # Related scenarios - representatives
        closest_representative = self.map_scenarios_to_representatives(
            representative_ids=[rep, new_rep],
            selected_scenario_ids=scenarios_rep,
            distance=distance_stage,
        )
        # Update the data
        self.update_tree(
            scenarios,
            Scen0,
            closest_representative,
            scenarios_rep,
            map_stages_columns,
            stage_id,
        )
        # Build the tree
        edges_related = self.filter_edges_by_vertex(graph, stage_id, rep)
        first_edge = edges_related[0]
        predecessor_id = first_edge[0]
        if (stage_id, rep) in tree.keys():
            edges_related = self.filter_edges_by_vertex(graph, stage_id, rep)
            first_edge = edges_related[0]
            predecessor_id = first_edge[0]
            current_id = first_edge[1]
            del tree[(stage_id, rep)]
            graph["edges"].remove(first_edge)
            del graph["ids"][current_id]
        if (stage_id, new_rep) in tree.keys():
            raise ValueError(f"({stage_id}, {new_rep}) should not be an existing key")
        for scen in scenarios_rep:
            rep = closest_representative[scen]
            key = (stage_id, rep)
            if key in tree.keys():
                tree[key].append(scen)
            else:
                tree[key] = [scen]
            if key not in graph["ids"].values():
                current_id = max(graph["ids"].keys()) + 1
                graph["ids"][current_id] = key
                graph["edges"].append((predecessor_id, current_id))
        # Update representatives list
        representatives_chosen_unsorted = list(set(closest_representative.values()))
        for rep in representatives_chosen_unsorted:
            if rep not in clusters:
                clusters.append(rep)
        representatives[stage_id] = clusters
        # update probabilities
        self.update_probability(prob_scenarios_stages, stage_ids, stage_id, tree, representatives)

    def update_graph(self, graph: Graph) -> None:
        """Update the graph that keeps the relationship between scenario IDs
        and their representatives.

        Args:
            graph (Graph): The graph.
        """
        # First, sort the graph
        graph_edges = graph["edges"]
        graph_edges_sorted = sorted(graph_edges, key=lambda x: x[0])
        graph["edges"] = graph_edges_sorted

        # Create a new mapping to ensure that the ids are consecutive.
        old_ids = graph["ids"]
        total_ids = len(old_ids.keys())
        new_numbering = [i for i in range(-1, total_ids)]
        mapping_ids = {i: j for i, j in zip(old_ids.keys(), new_numbering)}
        new_ids = {}
        for oid in old_ids.keys():
            new_ids[mapping_ids[oid]] = old_ids[oid]
        old_edges = graph["edges"]
        new_edges = []
        for oe in old_edges:
            new_edges.append((mapping_ids[oe[0]], mapping_ids[oe[1]]))
        graph["ids"] = new_ids
        graph["edges"] = new_edges
        return None

    def combine_tree_graph(self, graph: Graph, tree: TreeInfoMap) -> Tree:
        """
        Build a tree structure by combining a graph representation with scenario
        information.

        Args:
            graph (Graph): A graph structure.
            tree (TreeInfoMap): Mapping from (stage, representative) tuples to
                scenario ID lists.

        Returns:
            Tree: A list of nodes representing the constructed tree structure.
        """
        results = []
        new_map: Dict[int, Tuple[Optional[int], int]] = dict()
        counter_stage: Dict[Optional[int], int] = dict()
        ids = graph["ids"]
        for current_id, (stage, _) in ids.items():
            if stage in counter_stage.keys():
                counter = counter_stage[stage] + 1
            else:
                counter = 1
            counter_stage[stage] = counter
            new_map[current_id] = (stage, counter)
        edges = graph["edges"]
        for parent_id, child_id in edges:
            key = new_map[child_id]
            child_stage, child_representative = graph["ids"][child_id]
            assert child_stage is not None and child_representative is not None
            scenario_ids = tree[(child_stage, child_representative)]
            parent_key = new_map[parent_id]
            current_info: Node = {
                "key": key,
                "scenario_ids": scenario_ids,
                "parent_key": parent_key if parent_key[0] is not None else None,
                "description": f"Stage {key[0]}, cluster {key[1]}",
            }
            results.append(current_info)
        return results

    def generate_trees(
        self, r: float, initial_stage_id_to_cluster: Optional[int] = None
    ) -> List[Tree]:
        """Build the scenario tree.

        Args:
            r (float): Exponent used in the weighted norm computation.
            initial_stage_id_to_cluster (Optional[int]): Stage ID from which clustering starts.
                Default to None, which means that no clustering is performed.

        Raises:
            ValueError: If `initial_stage_id_to_cluster` is not a valid stage_id.
            ValueError: If `r` is negative or zero.
        Returns:
            List[Tree]: The trees.
        """
        # If initial_stage_id_to_cluster is None means that the data of all
        # stages have been observed, i.e., no clustering is performed.
        results: List[Tree] = []
        # Initial checks
        if (
            initial_stage_id_to_cluster is not None
            and initial_stage_id_to_cluster not in self.stage_ids
        ):
            raise ValueError(
                """Invalid value for `initial_stage_id_to_cluster`. It sholg be a value inside
                inside the list `stages_id`
                """
            )
        if self.num_scenarios is None:
            raise ValueError("Provide a value for `num_scenarios` when creating the object")
        if self.num_trees is None:
            raise ValueError("`num_trees` must be provided")
        if r <= 0:
            raise ValueError("`r` must be greater than 0")
        # Initial parameters
        num_random_variables = sum(self.num_variables_per_stage)
        # Get the initial column position and the last column position of the data for each stage
        map_stages_columns = self.mapping_stages_columns()
        stage_ids_with_initial_stage = list(map_stages_columns.keys())
        for i_tree in range(self.num_trees):
            graph: Graph = {"edges": [], "ids": {}}
            representatives: Dict[int, List[int]] = {}
            Scen0 = np.full(
                shape=(self.num_scenarios, num_random_variables),
                fill_value=np.nan,
                dtype=np.float64,
            )
            tree_information: TreeInfoMap = dict()
            current_scenarios = self.scenarios[i_tree]
            prob_scenarios_stages = np.full(
                shape=(self.num_scenarios, len(stage_ids_with_initial_stage)),
                fill_value=0.0,
                dtype=np.float64,
            )
            probability_scenarios = np.full(
                shape=self.num_scenarios, fill_value=1 / self.num_scenarios
            )
            threshold_stages = self.compute_stages_thresholds(
                current_scenarios,
                map_stages_columns,
                stage_ids_with_initial_stage,
                probability_scenarios,
                r,
                initial_stage_id_to_cluster,
            )
            # To begin with, the first clusters are created. The cration process is based on
            # the initial stage to cluster
            self.update_non_clustering_stages(
                current_scenarios,
                map_stages_columns,
                stage_ids_with_initial_stage,
                prob_scenarios_stages,
                initial_stage_id_to_cluster,
                tree_information,
                representatives,
                Scen0,
                graph,
            )
            # Create clusters for each stage. The distance obtained in each stage must be lower than
            # a threshold. Therefore, in each cluster, the distance will be lower
            # than the threshold. Then, new cluster will be created until the total distance
            # in the stage is lower than a threshold.
            if initial_stage_id_to_cluster is not None:
                idx_stage_id_initial_cluster = stage_ids_with_initial_stage.index(
                    initial_stage_id_to_cluster
                )
                stage_ids_to_complete = stage_ids_with_initial_stage[idx_stage_id_initial_cluster:]
                threshold_norm_stage = None
                stage_norm = None
                for stage_id in stage_ids_to_complete:
                    representatives_stage = []
                    # Get all the clusters from the predecessor stage
                    idx_stage_id = stage_ids_with_initial_stage.index(stage_id)
                    precedessor_stage_id = stage_ids_with_initial_stage[idx_stage_id - 1]
                    predecessor_clusters = representatives[precedessor_stage_id]
                    threshold_norm_stage = threshold_stages[stage_id]
                    # Compute the distance matrix of the current stage
                    distance_stage = self.compute_distance_stage(
                        current_scenarios, stage_id, map_stages_columns
                    )
                    # For a predecessor cluster, find representatives for this stage
                    # until the norm is lower than the threshold.
                    for p_cluster in predecessor_clusters:
                        p_cluster_norm = None
                        representatives_p_cluster = []
                        predecessor_key = (precedessor_stage_id, p_cluster)
                        predecessor_id = self.get_vertex_id(graph, precedessor_stage_id, p_cluster)
                        predecessor_scenarios = tree_information[predecessor_key]
                        remaining_scenarios = copy(predecessor_scenarios)
                        # Create the new clusters from p_cluster
                        closest_representative = dict()
                        while (p_cluster_norm is None) or (
                            p_cluster_norm is not None and p_cluster_norm > threshold_norm_stage
                        ):
                            # Select a cluster with more than 1 scenario
                            rep = self.get_representative(
                                distance_stage, probability_scenarios, remaining_scenarios, r
                            )
                            # Add it as a representative of the cluster
                            representatives_p_cluster.append(rep)
                            # Remove the representative from the list of scenarios
                            # to avoid choosing it again
                            idx_rep = remaining_scenarios.index(rep)
                            remaining_scenarios.pop(idx_rep)
                            # Relate scenarios - representatives
                            closest_representative = self.map_scenarios_to_representatives(
                                representatives_p_cluster, predecessor_scenarios, distance_stage
                            )
                            # Build the tree
                            self.update_tree(
                                current_scenarios,
                                Scen0,
                                closest_representative,
                                predecessor_scenarios,
                                map_stages_columns,
                                stage_id,
                            )
                            # Compute the norm of the tree
                            p_cluster_norm = self.compute_delta_norm_tree(
                                current_scenarios,
                                map_stages_columns,
                                stage_id,
                                Scen0,
                                probability_scenarios,
                                r,
                                predecessor_scenarios,
                            )
                        representatives_chosen_unsorted = list(set(closest_representative.values()))
                        representatives_chosen = [
                            x
                            for x in representatives_p_cluster
                            if x in representatives_chosen_unsorted
                        ]
                        representatives_stage.extend(representatives_chosen)
                        # Associate scenarios to each cluster
                        current_id = max(graph["ids"].keys()) + 1
                        for scen in predecessor_scenarios:
                            rep = closest_representative[scen]
                            key = (stage_id, rep)
                            if key in tree_information.keys():
                                tree_information[key].append(scen)
                            else:
                                tree_information[key] = [scen]
                            if key not in graph["ids"].values():
                                graph["ids"][current_id] = key
                                graph["edges"].append((predecessor_id, current_id))
                                current_id += 1
                    representatives[stage_id] = representatives_stage
                    # Update the values of the matrix prob_scenarios_stages
                    self.update_probability(
                        prob_scenarios_stages,
                        stage_ids_with_initial_stage,
                        stage_id,
                        tree_information,
                        representatives,
                    )
                    # Before moving to the next stage, compute the norm of the current stage.
                    # If the norm is greater than the threshold, keep on creating more clusters
                    stage_norm = self.compute_delta_norm_tree(
                        current_scenarios,
                        map_stages_columns,
                        stage_id,
                        Scen0,
                        probability_scenarios,
                        r,
                    )
                    # It might happen that that the norm of the stage is greater than the threshold.
                    # If so, we should keep refining.
                    if threshold_norm_stage is None:
                        raise ValueError("Unable to find a threshold value")
                    while stage_norm > threshold_norm_stage:
                        # Select a cluster with more than 1 scenario
                        # Split the cluster by means of selecting a new representative.
                        # Related scenarios - representatives
                        # Update the data
                        # update_probability
                        self.update_stage(
                            current_scenarios,
                            map_stages_columns,
                            stage_ids_with_initial_stage,
                            stage_id,
                            representatives,
                            tree_information,
                            graph,
                            Scen0,
                            prob_scenarios_stages,
                            distance_stage,
                        )
                        # Compute the norm of the stage
                        stage_norm = self.compute_delta_norm_tree(
                            current_scenarios,
                            map_stages_columns,
                            stage_id,
                            Scen0,
                            probability_scenarios,
                            r,
                        )
            self.update_graph(graph)
            combined_tree = self.combine_tree_graph(graph, tree_information)
            results.append(combined_tree)
        return results
