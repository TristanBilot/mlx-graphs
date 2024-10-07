from typing import Sequence, Union

import mlx.core as mx
import numpy as np

from mlx_graphs.data import GraphData, GraphDataBatch, batch
from mlx_graphs.datasets.dataset import Dataset


class Dataloader:
    """
    Default data loader to batch and iterate over multiple graphs.

    Args:
        dataset: `Dataset` or list of `GraphData` to batch and iterate over
        batch_size: Number of graphs to load per batch. Defaults to 1
        shuffle: Whether to reshuffle the order of graphs within each batch.
            Defaults to False
    """

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_size: int = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._indices = list(range(len(dataset)))
        self._current_index = 0

    def _shuffle_indices(self):
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self) -> GraphDataBatch:
        """
        Get the next batch of data.

        Returns:
            A batch of graphs.
        """
        if self._current_index >= len(self.dataset):
            self._current_index = 0
            if self.shuffle:
                self._shuffle_indices()
            raise StopIteration

        batch_indices = self._indices[
            self._current_index : self._current_index + self.batch_size
        ]
        batched_data = batch(
            [
                self.dataset[i]  # type: ignore - this is a list[GraphData]
                for i in batch_indices
            ]
        )
        batched_data.batch_size = batched_data._batch_indices.max().item() + 1

        self._current_index += self.batch_size
        return batched_data


class PaddedDataloader:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[GraphData]],
        batch_num_edges: int,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.batch_num_edges = batch_num_edges
        self.shuffle = shuffle

        self._indices = list(range(len(dataset)))
        self._current_index = 0

    def _shuffle_indices(self):
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self) -> GraphDataBatch:
        if self._current_index >= len(self.dataset):
            self._current_index = 0
            if self.shuffle:
                self._shuffle_indices()
            raise StopIteration

        cumsum_edges = 0
        end_index = 0
        for i in list(range(self._current_index, len(self._indices))):
            idx = self._indices[i]
            num_edges = self.dataset[idx].num_edges
            if cumsum_edges + num_edges <= self.batch_num_edges:
                cumsum_edges += num_edges
            else:
                end_index = i
                break

        # Handle the last batch
        if end_index == 0:
            end_index = len(self._indices)

        batch_indices = self._indices[self._current_index : end_index]
        if len(batch_indices) == 0:
            raise ValueError("Batch size should be larger")

        batched_data = batch(
            [
                self.dataset[i]  # type: ignore - this is a list[GraphData]
                for i in batch_indices
            ]
        )
        # Append no-op edges to pad the batched graph
        num_dummy_edges = self.batch_num_edges - batched_data.num_edges
        dummy_edges = mx.full((2, num_dummy_edges), -1)

        batched_data.edge_index = mx.concatenate(
            [batched_data.edge_index, dummy_edges], axis=1
        )

        # Append a no-op node feature
        if batched_data.node_features is not None:
            dummy_node_features = mx.zeros((1, batched_data.num_node_features))
            batched_data.node_features = mx.concatenate(
                [batched_data.node_features, dummy_node_features], axis=0
            )

        batched_data._batch_indices = mx.concatenate(
            [batched_data._batch_indices, mx.array([-1])], axis=0
        )
        batched_data.batch_size = batched_data._batch_indices.max().item() + 1

        self._current_index = end_index
        return batched_data
