import mlx.core as mx
import numpy as np
import pytest

from mlx_graphs.data import GraphData
from mlx_graphs.sampling import sample_neighbors
from mlx_graphs.sampling.neighbor_sampler import sample_nodes


def test_sample_nodes_one_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[2], input_node=0
    )

    assert sampled_edge.shape == (2, 2), "sampled edges shape is incorrect"
    assert len(n_id) == 3, "Incorrect reference to sampled nodes"
    assert len(e_id) == 2, "Incorrect reference to sampled edges"
    assert input_node == 0, "Incorrect reference to input nodes"


def test_sample_nodes_two_hop():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )

    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[3, 2], input_node=0
    )

    assert sampled_edge.shape == (2, 9), "sampled edges shape is incorrect"
    assert len(n_id) == 10, "Incorrect reference to sampled nodes"
    assert len(e_id) == 9, "Incorrect reference to sampled edges"
    assert input_node == 0, "Incorrect reference to input nodes"


def test_sample_nodes_edge_cases():
    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[0], input_node=0
    )

    assert np.array_equal(sampled_edge, np.array([]))
    assert np.array_equal(n_id, np.array([0]))
    assert np.array_equal(e_id, np.array([]))
    assert input_node == 0

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[-1], input_node=0
    )

    assert np.array_equal(sampled_edge, np.array([[0, 0, 0], [1, 2, 3]]))
    assert np.array_equal(n_id, np.array([0, 1, 2, 3]))
    assert len(e_id) == 3
    assert input_node == 0

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[2, -1], input_node=0
    )

    assert sampled_edge.shape == (2, 6)
    assert len(n_id) == 7
    assert len(e_id) == 6
    assert input_node == 0

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[1], input_node=100
    )

    assert np.array_equal(sampled_edge, np.array([]))
    assert np.array_equal(n_id, np.array([100]))
    assert np.array_equal(e_id, np.array([]))
    assert input_node == 100

    edge_index_numpy = np.array(
        [
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 1, 2, 2, 3, 3],
        ]
    )
    sampled_edge, n_id, e_id, input_node = sample_nodes(
        edge_index=edge_index_numpy, num_neighbors=[100], input_node=0
    )

    assert sampled_edge.shape == (2, 3)
    assert n_id == [0, 1, 2, 3]
    assert len(e_id) == 3
    assert input_node == 0


def test_sample_neighbors():
    # Default args
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    first_subgraph_edge_index = mx.array([[0, 0, 1, 1], [2, 3, 3, 4]])
    graph = GraphData(edge_index=edge_index)
    subgraphs = sample_neighbors(graph=graph, num_neighbors=[-1], input_nodes=[0, 1])
    assert len(subgraphs) == 1
    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(first_subgraph_edge_index)
    )

    # Test for batch_size 1 and 2 seed nodes
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    first_subgraph_edge_index = mx.array([[0, 0, 2, 3], [2, 3, 5, 6]])
    second_subgraph_edge_index = mx.array([[1, 1, 3, 4], [3, 4, 6, 7]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    subgraphs = sample_neighbors(
        graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=1
    )

    assert len(subgraphs) == 2, "Incorrect length of subgraphs"
    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(first_subgraph_edge_index)
    ), "Incorrect sampling"
    assert mx.array_equal(
        mx.sort(subgraphs[1].edge_index), mx.sort(second_subgraph_edge_index)
    ), "Incorrect sampling"

    # Test for batch_size 2 and 2 seed nodes
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    subgraph_edge_index = mx.array([[0, 0, 2, 3, 1, 1, 3, 4], [2, 3, 5, 6, 3, 4, 6, 7]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    subgraphs = sample_neighbors(
        graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=2
    )
    assert len(subgraphs) == 1, "Incorrect length of subgraphs"
    assert mx.array_equal(
        mx.sort(subgraphs[0].edge_index), mx.sort(subgraph_edge_index)
    ), " Incorrect sampling"

    # test for incorrect inputs
    edge_index = mx.array([[0, 0, 1, 1, 2, 3, 4], [2, 3, 3, 4, 5, 6, 7]])
    node_features = mx.array([[1], [1], [1], [1], [1], [1], [1], [1]])
    graph = GraphData(edge_index=edge_index, node_features=node_features)
    with pytest.raises(ValueError):
        subgraphs = sample_neighbors(
            graph=graph, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=0
        )
    with pytest.raises(ValueError):
        subgraphs = sample_neighbors(
            graph=graph, input_nodes=[], num_neighbors=[2, 1], batch_size=0
        )
    with pytest.raises(ValueError):
        subgraphs = sample_neighbors(
            graph=graph, input_nodes=[0, 1], num_neighbors=[], batch_size=1
        )
    with pytest.raises(ValueError):
        subgraphs = sample_neighbors(
            graph=edge_index, input_nodes=[0, 1], num_neighbors=[2, 1], batch_size=1
        )
