from typing import List, TypedDict

import networkx as nx

_TRANSITION_RELATIONS = {
    "<": {
        "<": "<",
        "=": "<",
        "-": "-",
    },
    "=": {
        "<": "<",
        "=": "=",
        "-": "-",
    },
    "-": {
        "<": "-",
        "=": "-",
        "-": "-",
    },
}


class _DictRelation(TypedDict):
    source: str
    target: str
    relation: str


def compute_temporal_closure(relations: List[_DictRelation]):
    """
    Compute the temporal closure of a set of temporal relations.

    This algorithm performs the following steps:
    1. Create a directed graph from the input relations.
    2. Infer new relations by traversing the graph and applying transition rules.
    3. Combine inferred relations with single-hop relations.

    The algorithm uses depth-first search (DFS) to explore paths between nodes
    and applies the transition rules defined in _TRANSITION_RELATIONS to infer
    new relations along these paths.

    Args:
        relations (List[_DictRelation]): A list of dictionaries representing
            temporal relations. Each dictionary contains 'source', 'target',
            and 'relation' keys.

    Returns:
        List[_DictRelation]: A list of dictionaries representing the temporal
        closure, including both original and inferred relations.

    Note:
        - The '>' relation is converted to '<' by swapping source and target.
        - The algorithm handles three types of relations: '<', '=', and '-'.
        - Inferred relations are determined using the _TRANSITION_RELATIONS table.
    """
    graph = nx.DiGraph()

    for relation in relations:
        source, target, rel_type = (
            relation["source"],
            relation["target"],
            relation["relation"],
        )
        if rel_type == ">":
            graph.add_edge(target, source, relation="<")
        else:
            graph.add_edge(source, target, relation=rel_type)

    # Infer relations using _TRANSITION_RELATIONS
    inferred_relations = set()
    for source in graph.nodes():
        for target in nx.dfs_preorder_nodes(graph, source=source):
            if source == target:
                continue

            path = nx.shortest_path(graph, source, target)
            current_relation = graph[path[0]][path[1]]["relation"]

            for i in range(2, len(path)):
                next_node = path[i]
                next_relation = graph[path[i - 1]][next_node]["relation"]
                inferred_relation = _TRANSITION_RELATIONS[current_relation][
                    next_relation
                ]
                inferred_relations.add((source, inferred_relation, next_node))
                current_relation = inferred_relation

    # Add single-hop relations to the inferred set
    for edge in graph.edges(data=True):
        inferred_relations.add((edge[0], edge[2]["relation"], edge[1]))

    return [
        {"source": source, "target": target, "relation": relation}
        for source, relation, target in inferred_relations
    ]
