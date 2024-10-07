import time

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

DATASET = "Cora"
NUM_NEIGHBORS = [10, 10]
BATCH_SIZE = 1
dataset = Planetoid(root=".", name=DATASET)
graph = dataset[0]
input_nodes = list(range(graph.num_nodes))


if __name__ == "__main__":
    start_time = time.time()
    loader = NeighborLoader(
        dataset[0],
        input_nodes=torch.tensor(input_nodes),
        num_neighbors=[10, 10],
        batch_size=1,
        replace=False,
        shuffle=False,
    )
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Execution time: {total_time} seconds")
