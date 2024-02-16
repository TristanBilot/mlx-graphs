.. _quickstart:


Quickstart
============

Here is a short introduction to ``mlx-graphs``, starting from building a simple graph to training a first GNN model.

We'll cover the following topics:

.. contents::
    :local:


Representing Graphs
--------------------

Simple Graphs
~~~~~~~~~~~~~

A graph can be easily built in ``mlx-graphs`` using :class:`~mlx_graphs.data.data.GraphData`.

.. code-block:: python

    import mlx.core as mx
    from mlx_graphs.data.data import GraphData

    edge_index = mx.array([[0, 1, 0, 2, 3],
                           [2, 3, 1, 0, 2]])
    node_features = mx.random.normal((4, 8))

    graph = GraphData(edge_index, node_features)
    >>> GraphData(
            edge_index(shape=(2, 5), int32)
            node_features(shape=(4, 8), float32))

    graph.edge_index
    >>> array([[0, 1, 0, 2, 3],
               [2, 3, 1, 0, 2]], dtype=int32)


The :class:`~mlx_graphs.data.data.GraphData` class accepts by default the following optional arguments to build the graph.

``edge_index``:
    an array of size ``[2, num_edges]`` which specifies the topology of the graph in `COO <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html>`_ format. The i-th column in ``edge_index`` defines the source and destination nodes of the i-th edge
``node_features``:
    an array of size ``[num_nodes, num_node_features]`` defining the features associated to each node (if any). The i-th row contains the features of the i-th node
``edge_features``:
     an array of size ``[num_edges, num_edge_features]`` defining the features associated to each edge (if any). The i-th row contains the features of the i-th edge
``graph_features``:
    an array of size ``[1, num_graph_features]`` defining the features associated to the graph itself
``node_labels``:
    an array of shape ``[num_nodes, num_node_labels]`` containing the labels of each node.
``edge_labels``:
    an array of shape ``[num_edges, num_edge_labels]`` containing the labels of each edge.
``graph_labels``
    an array of shape ``[1, num_graph_labels]`` containing the global labels of the graph.

We adopt the above convention across the entire library both in terms of shapes of the attributes and the order in which they're provided to functions.

Graphs with Custom Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mlx_graphs.data.data.GraphData` also supports additional attributes to integrate within the graph.
Just add your own attributes to the constructor using ``kwargs``.

.. code-block:: python

    import mlx.core as mx
    from mlx_graphs.data.data import GraphData

    graph = GraphData(
        edge_index=mx.array([[0, 0, 0], [1, 1, 1]]),
        node_features=mx.random.normal((4, 8)),
        alpha=mx.ones((4,)),  # Custom attribute
    )
    >>> GraphData(
            edge_index(shape=(2, 3), int32)
            node_features(shape=(4, 8), float32)
            alpha(shape=(4,), float32))

    graph.alpha
    >>> array([1, 1, 1, 1], dtype=float32)


Operations on graphs
--------------------

We provide some utilities to perform operations on graphs in :mod:`~mlx_graphs.utils`.

For example, :meth:`~mlx_graphs.utils.transformations.to_edge_index`
and :meth:`~mlx_graphs.utils.transformations.to_adjacency_matrix` can be used to convert an
adjacency matrix representing a graph into its edge index and viceversa.


Batching
---------

In tasks with multiple graphs, such as graph classification, batching accelerates GNN computations by
processing several graphs together instead of individually. This approach can drastically enhance speed
through parallelization on the Mac's GPU.

The :class:`~mlx_graphs.data.batch.GraphDataBatch` class handles all batch operations, enabling the creation of a batch from a list of
:class:`~mlx_graphs.data.data.GraphData` objects. We provide the :meth:`mlx_graphs.data.batch.batch` function as an interface to create a :class:`~mlx_graphs.data.batch.GraphDataBatch` out of a sequence of :class:`~mlx_graphs.data.data.GraphData` objects.

.. hint::

    The operations provided in ``mlx-graphs`` are particularly efficient on large graphs. We recommend to leverage batching whenever possible,
    ensuring that the batched graphs collectively fit within available memory.


.. code-block:: python

    from mlx_graphs.data.batch import batch

    graphs = [
        GraphData(
            edge_index=mx.array([[0, 0, 0], [1, 1, 1]]),
            node_features=mx.zeros((3, 1)),
        ),
        GraphData(
            edge_index=mx.array([[1, 1, 1], [2, 2, 2]]),
            node_features=mx.ones((3, 1)),
        ),
        GraphData(
            edge_index=mx.array([[3, 3, 3], [4, 4, 4]]),
            node_features=mx.ones((3, 1)) * 2,
        )
    ]
    graphs_batch = batch(graphs)
    >>> GraphDataBatch(
        edge_index(shape=[2, 9], int32)
        node_features(shape=[9, 1], float32))

    graphs_batch.num_graphs
    >>> 3

Internally, :class:`~mlx_graphs.data.batch.GraphDataBatch` simply collates the attributes
from all :class:`~mlx_graphs.data.data.GraphData` objects and concatenates them to end up with a single large graph
made of the disconnected graphs. Our batching strategy follows a similar approach as in `PyG <https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#mini-batches>`_.

If needed, the graphs within the batch can be easily indexed:

.. code-block:: python

    graphs_batch[1]
    >>> GraphData(
        edge_index(shape=[2, 3], int32)
        node_features(shape=[3, 1], float32))

    graphs_batch[1:]
    >>> [
            GraphData(
                edge_index(shape=[2, 3], int32)
                node_features(shape=[3, 1], float32)),
            GraphData(
                edge_index(shape=[2, 3], int32)
                node_features(shape=[3, 1], float32))
        ]

Datasets and Data loaders
-------------------------

Datasets can be implemented by extending the base class :class:`~mlx_graphs.datasets.Dataset` and implementing its abstract methods. For example, a basic implementation of the QM7b molecular dataset could look like

.. code-block:: python

    import os

    import mlx.core as mx
    import scipy as sp

    from mlx_graphs.data import GraphData
    from mlx_graphs.datasets.dataset import Dataset
    from mlx_graphs.datasets.utils import download
    from mlx_graphs.utils.transformations import to_sparse_adjacency_matrix

    class QM7bDataset(Dataset):
        def __init__(self):
            super().__init__(name="qm7b")

        def download(self):
            file_path = os.path.join(self.raw_path, self.name + ".mat")
            download(
                "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm7b.mat",
                path=file_path,
            )

        def process(self):
            mat_path = os.path.join(self.raw_path, self.name + ".mat")
            data = scipy.io.loadmat(mat_path)
            labels = mx.array(data["T"].tolist())
            features = mx.array(data["X"].to_list())
            num_graphs = labels.shape[0]
            graphs = []
            for i in range(num_graphs):
                edge_index, edge_features = to_sparse_adjacency_matrix(features[i])
                graphs.append(
                    GraphData(
                        edge_index=edge_index,
                        edge_features=edge_features,
                        graph_labels=mx.expand_dims(labels[i], 0),
                    )
                )
            self.graphs = graphs

Every :class:`~mlx_graphs.datasets.Dataset` should implement two abstract methods: :attr:`~.mlx_graphs.datasets.Dataset.download`, responsible to download raw datasets locally
and :attr:`~.mlx_graphs.datasets.Dataset.process()`, which transforms the datasets into a ``List[GraphData]``, saved in ``self.graphs``.
Once the list of graphs is processed, all the indexing and dataset properties such as :attr:`~.mlx_graphs.datasets.Dataset.num_graphs`,
:attr:`~.mlx_graphs.datasets.Dataset.num_node_features` and :attr:`~.mlx_graphs.datasets.Dataset.num_node_classes` are automatically handled.


We provide a few widely used datasets and we expect to implement more over time.

Data loaders can be used to automatically load and batch graphs for training routines.
The :class:`~mlx_graphs.loaders.Dataloader` class accepts a :class:`~mlx_graphs.datasets.Dataset`
or a sequence of :class:`~mlx_graphs.data.data.GraphData` as input together with a ``batch_size``
and provides an iterable over the dataset.

.. code-block:: python

    from mlx_graphs.loaders import Dataloader

    data = QM7bDataset()
    loader = Dataloader(data, batch_size=16)

    for item in loader:
        ...



GNN Layers
------------

Similarly as other libraries, ``mlx-graphs`` comes with some predefined GNN layers. These layers usually follow the
implementation from the original papers and can be used as basic blocks to build more complex GNN models.

For instance, here is a :class:`~mlx_graphs.nn.SAGEConv` layer from the `Inductive Representation Learning on Large Graphs <https://arxiv.org/abs/1706.02216>`_ paper:

.. code-block:: python

    import mlx.core as mx
    from mlx_graphs.data.data import GraphData
    from mlx_graphs.nn import SAGEConv

    graph = GraphData(
        edge_index = mx.array([[0, 1, 2, 3, 4], [0, 0, 1, 1, 3]]),
        node_features = mx.ones((5, 16)),
    )

    conv = SAGEConv(node_features_dim=16, out_features_dim=32)
    h = conv(graph.edge_index, graph.node_features)

    >>> h
    array([[1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
        [1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
        [1.05823, -0.295776, 0.075439, ..., -0.104383, 0.031947, -0.351897],
        [1.65429, -0.376169, 1.04172, ..., -0.919106, 1.42576, 0.490938],
        [1.05823, -0.295776, 0.075439, ..., -0.104383, 0.031947, -0.351897]],
        dtype=float32)
