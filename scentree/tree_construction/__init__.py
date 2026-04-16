from typing import Dict, List, Optional, Tuple, TypedDict, Union


class Node(TypedDict):
    """
    Represents a node in a scenario tree structure.

    Each node is uniquely identified by a `key` and may reference a parent
    node through `parent_key`. It contains metadata describing which scenarios
    it belongs to and an optional textual description.

    Attributes:
        key (Tuple[int, int]): Unique identifier of the node in the tree.
        scenario_ids (List[int]): List of scenario identifiers associated
            with this node.
        parent_key (Union[Tuple[int, int], None]): Identifier of the parent node
            in the tree. None if the node is a root node.
        description (str): Human-readable description of the node.
    """

    key: Tuple[Optional[int], int]
    scenario_ids: List[int]
    parent_key: Union[Tuple[Optional[int], Optional[int]], None]
    description: str


# A tree is a list of nodes.
Tree = List[Node]

# Mapping for trees.
TreeInfoMap = Dict[Tuple[int, int], List[int]]
