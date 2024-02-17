import timeit

import dgl
import dgl.data as dgl_datasets
import dgl.nn.pytorch as dgl_nn
import mlx.core as mx
import torch
import torch._dynamo
import torch.optim
import torch_geometric.datasets as pyg_datasets
import torch_geometric.nn as pyg_nn
from dgl_setup import setup_training_dgl, train_dgl
from mlx_graphs_setup import setup_training_mxg, train_mxg
from pyg_setup import setup_training_pyg, train_pyg

import mlx_graphs.datasets as mxg_datasets
import mlx_graphs.nn as mxg_nn

torch._dynamo.config.suppress_errors = True
mx.set_default_device(mx.gpu)


# Benchmark
frameworks = ["dgl", "pyg", "mxg"]
datasets = ["BZR_MD", "MUTAG", "DD", "NCI-H23"]
layers = ["GCNConv", "GATConv"]

batch_size = 64
hid_size = 128

TIMEIT_REPEAT = 5
TIMEIT_NUMBER = 1
COMPILE = False

torch.manual_seed(42)
mx.random.seed(42)
dgl.seed(42)


framework_to_setup = {
    "mxg": setup_training_mxg,
    "pyg": setup_training_pyg,
    "dgl": setup_training_dgl,
}

framework_to_train = {
    "mxg": train_mxg,
    "pyg": train_pyg,
    "dgl": train_dgl,
}


def dgl_dataset(name):
    pyg_dataset = pyg_datasets.TUDataset(f".mlx_graphs_data/{name}", name)
    dgl_dataset = dgl_datasets.TUDataset(dataset_name)

    for i, (pyg_p, dgl_p) in enumerate(zip(pyg_dataset, dgl_dataset.graph_lists)):
        dgl_dataset.graph_lists[i].ndata["x"] = pyg_p.x

    return dgl_dataset


framework_to_datasets = {
    "mxg": lambda name: mxg_datasets.TUDataset(name),
    "pyg": lambda name: pyg_datasets.TUDataset(f".mlx_graphs_data/{name}", name),
    "dgl": lambda name: dgl_dataset(name),
}
layer_classes = {
    "mxg": {
        "GCNConv": mxg_nn.GCNConv,
        "GATConv": mxg_nn.GATConv,
    },
    "pyg": {
        "GCNConv": pyg_nn.GCNConv,
        "GATConv": pyg_nn.GATConv,
    },
    "dgl": {
        "GCNConv": dgl_nn.GraphConv,
        "GATConv": dgl_nn.GATConv,
    },
}


def benchmark(framework, loader, step, state):
    train_fn = framework_to_train[framework]
    train_fn(loader, step, state)


for dataset_name in datasets:
    print(dataset_name)
    print("=" * 10)

    for framework in frameworks:
        dataset = framework_to_datasets[framework](dataset_name)

        for i, layer_name in enumerate(layers):
            layer = layer_classes[framework][layer_name]
            loader, step, state = framework_to_setup[framework](
                dataset, layer, batch_size, hid_size, compile=COMPILE
            )

            times = timeit.Timer(
                lambda: benchmark(framework, loader, step, state)
            ).repeat(repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER)

            time = min(times) / TIMEIT_NUMBER

            print(
                " | ".join(
                    [
                        f"{framework}",
                        f"{layer_name}",
                        f"{time:.3f}s",
                    ]
                )
            )
        print("")
