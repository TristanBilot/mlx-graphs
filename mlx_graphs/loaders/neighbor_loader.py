from typing import List

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData
from mlx_graphs.sampling import sample_nodes


class NeighborLoader:
    def __init__(
        self, graph_data: GraphData, batch_size: int, num_neighbors: List[int]
    ):
        self.graph_data = graph_data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.all_nodes = np.arange(graph_data.node_features.shape[0])

    def __iter__(self):
        # Shuffle all nodes at the start of each epoch
        np.random.shuffle(self.all_nodes)
        for i in range(0, len(self.all_nodes), self.batch_size):
            batch_nodes = self.all_nodes[i : i + self.batch_size]
            edge_index = np.array(self.graph_data.edge_index)
            sampled_edges, n_id, e_id, input_nodes = sample_nodes(
                edge_index=edge_index,
                num_neighbors=self.num_neighbors,
                batch_size=len(batch_nodes),
                input_nodes=batch_nodes.tolist(),
            )

            # Extract node features for n_id
            n_id = [int(id) for id in n_id]
            node_features = self.graph_data.node_features[mx.array(n_id), :]

            # Construct and yield a new GraphData object for the batch
            batch_graph_data = GraphData(
                edge_index=mx.array(sampled_edges).T,
                node_features=node_features,
                n_id=mx.array(n_id),
                e_id=mx.array(e_id),
                input_nodes=mx.array(input_nodes),
            )
            yield batch_graph_data
