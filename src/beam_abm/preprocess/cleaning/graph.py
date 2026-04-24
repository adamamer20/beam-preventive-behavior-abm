from __future__ import annotations

from collections.abc import Callable

import networkx as nx

from beam_abm.common.logging import get_logger

from .models import (
    DataCleaningModel,
    ImputationTransformation,
    RemovalTransformation,
    RenamingTransformation,
    Transformation,
)
from .semantics import iter_dependency_source_ids

logger = get_logger(__name__)


class TransformationNode:
    def __init__(self, col_id: str, index_in_col: int, transformation: Transformation):
        self.col_id = col_id
        self.index_in_col = index_in_col
        self.transformation = transformation

    def __repr__(self):
        return f"Node(col={self.col_id}, idx={self.index_in_col}, type={self.transformation.type})"

    def __hash__(self):
        return hash((self.col_id, self.index_in_col, id(self.transformation)))


def build_transformation_graph(
    *,
    model: DataCleaningModel,
    resolve_ref: Callable[[str], str],
) -> list[TransformationNode]:
    logger.info("Building transformation dependency graph")
    graph = nx.DiGraph()
    column_nodes_map: dict[str, list[TransformationNode]] = {}
    prediction_nodes: list[TransformationNode] = []

    all_columns = list(model.old_columns)
    if model.new_columns is not None:
        all_columns.extend(model.new_columns)

    total_transformations = 0
    for col in all_columns:
        nodes_for_col: list[TransformationNode] = []
        if col.transformations:
            for i, transformation in enumerate(col.transformations):
                node = TransformationNode(col.id, i, transformation)
                nodes_for_col.append(node)
                graph.add_node(node)
                total_transformations += 1
                if (
                    isinstance(transformation, ImputationTransformation)
                    and transformation.imputation_value == "$PREDICTION"
                ):
                    prediction_nodes.append(node)
        column_nodes_map[col.id] = nodes_for_col

    logger.info(f"Created {total_transformations} transformation nodes, {len(prediction_nodes)} prediction nodes")

    sequential_edges = 0
    for nodes_list in column_nodes_map.values():
        nodes_list.sort(key=lambda n: n.index_in_col)
        for idx in range(len(nodes_list) - 1):
            graph.add_edge(nodes_list[idx], nodes_list[idx + 1])
            sequential_edges += 1
    logger.debug(f"Added {sequential_edges} sequential edges within columns")

    def last_non_removal_node_for_column(col_id: str) -> TransformationNode | None:
        if not column_nodes_map[col_id]:
            return None
        non_removal_nodes = [
            node
            for node in column_nodes_map[col_id]
            if not isinstance(node.transformation, RemovalTransformation | RenamingTransformation)
        ]
        if not non_removal_nodes:
            return None
        return max(non_removal_nodes, key=lambda n: n.index_in_col)

    def removal_node_for_column(col_id: str) -> TransformationNode | None:
        for node in column_nodes_map[col_id]:
            if isinstance(node.transformation, RemovalTransformation | RenamingTransformation):
                return node
        return None

    dependency_edges = 0
    for nodes_list in column_nodes_map.values():
        for node in nodes_list:
            transformation = node.transformation
            source_ids = iter_dependency_source_ids(transformation, resolve_ref=resolve_ref)
            for src_col_id in source_ids:
                last_src = last_non_removal_node_for_column(src_col_id)
                if last_src:
                    graph.add_edge(last_src, node)
                    dependency_edges += 1
                source_removal_node = removal_node_for_column(src_col_id)
                if source_removal_node:
                    graph.add_edge(node, source_removal_node)
                    dependency_edges += 1

    logger.debug(f"Added {dependency_edges} dependency edges")

    last_node_in_column: dict[str, TransformationNode] = {}
    for col_id, nodes_list in column_nodes_map.items():
        if nodes_list:
            last_node_in_column[col_id] = max(nodes_list, key=lambda n: n.index_in_col)

    forced_edges = []
    cycle_prevented_edges = 0
    for prediction_node in prediction_nodes:
        for last_node in last_node_in_column.values():
            if last_node == prediction_node:
                continue
            graph.add_edge(last_node, prediction_node)
            try:
                nx.find_cycle(graph, source=prediction_node)
                graph.remove_edge(last_node, prediction_node)
                cycle_prevented_edges += 1
            except nx.NetworkXNoCycle:
                forced_edges.append((last_node, prediction_node))

    logger.info(f"Added {len(forced_edges)} forced edges, prevented {cycle_prevented_edges} cycles")

    try:
        sorted_nodes = list(nx.topological_sort(graph))
        logger.info(f"Successfully created topological sort with {len(sorted_nodes)} nodes")
        return sorted_nodes
    except nx.NetworkXUnfeasible as error:
        logger.error("Cycle detected while performing topological sort")
        try:
            cycle = nx.find_cycle(graph)
            logger.error(f"Detected cycle: {cycle}")
        except nx.NetworkXNoCycle:
            logger.error("Could not identify specific cycle")
        raise ValueError("Cycle detected while performing topological sort.") from error
