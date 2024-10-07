import logging
import time

from line_profiler import LineProfiler

from mlx_graphs.datasets import PlanetoidDataset
from mlx_graphs.sampling import sample_neighbors
from mlx_graphs.sampling.neighbor_sampler import sample_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
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
    start_time = time.time()

    sample_neighbors(graph, NUM_NEIGHBORS, BATCH_SIZE, input_nodes)

    end_time = time.time()

    total_time = end_time - start_time
    logger.info(f"Sampler execution time: {total_time} seconds")


if __name__ == "__main__":
    lp = LineProfiler()

    lp.add_function(sample_neighbors)

    lp.add_function(sample_nodes)

    lp_wrapper = lp(main)

    lp_wrapper()

    lp.print_stats()
