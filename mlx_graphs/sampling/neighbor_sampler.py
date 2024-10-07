from typing import Sequence, Tuple

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData


def sample_neighbors(
    graph: GraphData,
    num_neighbors: list[int],
    batch_size: int = None,
    input_nodes: list[int] = None,
) -> list[GraphData]:
    """
    Samples subgraphs from a graph following a neighbor sampling strategy.

    By default, each node samples its neighborhood to keep only a fixed number
    of neighbors. To apply the sampling only on specific nodes, one can set
    the ID of the given nodes in ``input_nodes``.

    To control how many node subgraphs to merge together into a unique graph,
    ``batch_size`` can be set accordingly. For example, sampling all nodes
    from a 100-nodes graph with a batch_size of 10 will yield a list of 10
    graphs, where each graph contains 10 nodes with their sampled neighborhood.

    To control the size of the neighborhood to sample, ``num_neighbors`` specifies
    how many neighbors to keep for each hop. For example, when set to `[40, 20]`,
    each node keeps maximum of 40 nodes for the first hop and 20 nodes for the second.

    Args:
        graph: The graph to sample
        num_neighbors: A list where each element represents the number of nodes to
            sample for each node at each hop.
        batch_size: The number of subgraphs to merge toegether. By default, no
            batching happens, a single graph with all the sampled nodes is returned.
        input_nodes: The specific nodes on which apply the sampling. By default,
            all nodes are considered in the sampling.

    Returns:
        A list of subgraphs, where each subgraph contains the sampled nodes, with
        a variying size depending on ``batch_size`` and ``num_neighbors``.

    Example:

    .. code-block:: python

        edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
        graph = GraphData(edge_index=edge_index)

        #  Sample nodes 0 and 1 with two neighbors
        subgraphs = sample_neighbors(
            graph=graph, num_neighbors=[-1], input_nodes=[0, 1]
        )
        >>> [
            GraphData(
                edge_index(shape=(2, 4), int64)
                n_id(shape=(6,), int64)
                e_id(shape=(2,), int32)
                input_nodes(shape=(), int32))
            ]

        subgraphs[0].edge_index
        >>> array([[0, 0, 1, 1],
                   [2, 3, 3, 4]], dtype=int64)


        #  Sample all nodes with only one neighbor
        subgraphs = sample_neighbors(
            graph=graph, num_neighbors=[1]
        )
        >>> [
            GraphData(
                edge_index(shape=(2, 5), int64)
                n_id(shape=(13,), int64)
                e_id(shape=(0,), float32)
                input_nodes(shape=(), int32))
            ]

        subgraphs[0].edge_index
        >>> array([[0, 1, 2, 3, 4],
                   [2, 4, 5, 6, 7]], dtype=int64)


        #  Same process, but we batch 3 graphs together instead of all graphs
        subgraphs = sample_neighbors(
            graph=graph, num_neighbors=[1], batch_size=3,
        )
        >>> [
            GraphData(
                edge_index(shape=(2, 3), int64)
                n_id(shape=(6,), int64)
                e_id(shape=(1,), int32)
                input_nodes(shape=(), int32)),
            GraphData(
                edge_index(shape=(2, 2), int64)
                n_id(shape=(5,), int64)
                e_id(shape=(0,), float32)
                input_nodes(shape=(), int32))
            ]

        subgraphs[0].edge_index
        >>> array([[0, 1, 2],
                   [3, 3, 5]], dtype=int64)

        subgraphs[1].edge_index
        >>> array([[3, 4],
                   [6, 7]], dtype=int64)
    """

    if not isinstance(num_neighbors, Sequence) or len(num_neighbors) == 0:
        raise ValueError(
            "Argument `num_neighbors` should be a non-empty list of integers"
        )

    if not isinstance(graph, GraphData):
        raise ValueError("graph must be a GraphData object")

    if not isinstance(num_neighbors, list):
        try:
            num_neighbors = list(num_neighbors)
        except TypeError:
            raise ValueError("num_neighbors must be a list.")

    input_nodes = (
        input_nodes if input_nodes is not None else list(range(graph.num_nodes))
    )
    batch_size = batch_size if batch_size is not None else graph.num_nodes

    batched_graphs = []

    for i in range(0, len(input_nodes), batch_size):
        batch_nodes = input_nodes[i : i + batch_size]

        # lists to store aggregated values for the current batch
        batch_sampled_edges = np.empty([2, 0], dtype=int)
        batch_n_id = np.empty([1, 0], dtype=int)
        batch_e_id = []
        batch_input_node = []

        for seed_node in batch_nodes:
            sampled_edges, n_id, e_id, input_node = sample_nodes(
                edge_index=np.array(graph.edge_index),
                num_neighbors=num_neighbors,
                input_node=seed_node,
            )
            if sampled_edges.size == 0:
                sampled_edges = np.empty([2, 0], dtype=int)
            batch_sampled_edges = np.append(batch_sampled_edges, sampled_edges, axis=1)
            batch_n_id = np.append(batch_n_id, n_id)
            batch_e_id.extend(e_id)
            batch_input_node.append(input_node)

        if batch_sampled_edges.any():
            subgraph = GraphData(
                edge_index=mx.array(batch_sampled_edges),
                n_id=mx.array(batch_n_id),
                e_id=mx.array(e_id),
                input_nodes=mx.array(input_node),
            )
            if graph.node_features is not None:
                subgraph.node_features = graph.node_features[mx.array(batch_n_id)]

            batched_graphs.append(subgraph)

    return batched_graphs


def sample_nodes(
    edge_index: np.ndarray, num_neighbors: list[int], input_node: int
) -> Tuple[np.array, list[int], list[int], int]:
    """GraphSage neighbor sampler implementation.

    Args:
        edge_index : the edge index representing the original graph
        to sample from nodes.
        num_neighbors : number of neighbors to sample for each hop.
        input_nodes : seed_nodes to take into consideration.

    Returns:
        sampled_edges_array : the sampled edge_index.
        n_id_list : a reference to the sampled nodes.
        e_id : a reference to the indices of sampled edges.
        input_nodes : a reference to the seed nodes.

    """
    sampled_edges = []
    sampled_edges_indices = []
    n_id = set([input_node])  # Will include all unique nodes encountered
    visited_nodes = set([input_node])
    current_layer_nodes = np.array([input_node], dtype=np.int32)

    for num_hop, num in enumerate(num_neighbors):
        next_layer_nodes = set()
        for node in current_layer_nodes:
            # Find indices of outgoing edges from the current node
            indices = np.where(edge_index[0] == node)[0]
            possible_targets = edge_index[1][indices]

            # Filter out already visited nodes
            targets = np.setdiff1d(
                possible_targets, np.array(list(visited_nodes), dtype=np.int32)
            )

            if len(targets) > 0:
                if num == -1:
                    selected_targets = targets
                    selected_indices = indices[: len(targets)]
                else:
                    num_edges_to_select = min(len(targets), num)
                    selected_targets_idx = np.random.choice(
                        range(len(targets)), size=num_edges_to_select, replace=False
                    )
                    selected_targets = targets[selected_targets_idx]
                    selected_indices = indices[selected_targets_idx]

                next_layer_nodes.update(selected_targets)
                visited_nodes.update(selected_targets)
                sampled_edges_indices.extend(selected_indices.tolist())

                for target in selected_targets:
                    sampled_edges.append([node, target])

        current_layer_nodes = np.array(list(next_layer_nodes - n_id), dtype=np.int32)
        n_id.update(next_layer_nodes)

    sampled_edges_array = np.array(sampled_edges).T
    n_id_list = list(n_id)

    return sampled_edges_array, n_id_list, sampled_edges_indices, input_node
