import cProfile
import logging
import os
import time

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.sampling import sample_neighbors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("application.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


DATASET = "pubmed"
NUM_NEIGHBORS = [10, 10]
BATCH_SIZE = 1
dataset = PlanetoidDataset(DATASET)
graph = dataset[0]
input_nodes = list(range(graph.num_nodes))

num_nodes = graph.num_nodes
num_edges = graph.edge_index.shape[1]


def main():
    logger.info(f"Number of nodes in graph: {num_nodes}")
    logger.info(f"Number of edges in graph: {num_edges}")

    start_time = time.time()

    subgraphs = sample_neighbors(graph, NUM_NEIGHBORS, BATCH_SIZE, input_nodes)

    end_time = time.time()

    total_time = end_time - start_time
    logger.info(f"Sampler execution time: {total_time} seconds")
    logger.info(f"sampled nodes: {subgraphs[0].n_id}")
    logger.info(f"number of edges sampled: {subgraphs[0].edge_index.shape[1]}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    profiling_dir = os.path.join(script_dir, "profiling_files")
    os.makedirs(profiling_dir, exist_ok=True)

    profile_filename = os.path.join(
        profiling_dir,
        f"sampling_{DATASET}_{NUM_NEIGHBORS}_{BATCH_SIZE}.prof",
    )

    cProfile.run("main()", profile_filename)
