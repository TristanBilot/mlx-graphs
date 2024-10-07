import platform
import timeit
from importlib.metadata import version

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
from tqdm import tqdm
from utils import to_markdown_table

import mlx_graphs.datasets as mxg_datasets
import mlx_graphs.nn as mxg_nn

mx.set_default_device(mx.gpu)

torch.manual_seed(42)
mx.random.seed(42)
dgl.seed(42)

# config
BATCH_SIZE = 64
HIDDEN_SIZE = 128

EPOCHS = 1  # epochs to run the training loop for
TIMEIT_REPEAT = 10  # repeat to use in timeit
TIMEIT_NUMBER = 1  # iterations per repeat in timeit
COMPILE = {  # whether to compile training step
    "mxg": [False, True],
    "pyg": [False],
    "dgl": [False],
}

FRAMEWORKS = [
    # "dgl",
    # "pyg",
    "mxg",
]
DATASETS = [
    "BZR_MD",
    "MUTAG",
    "DD",
    "NCI-H23",
]
LAYERS = [
    "GCNConv",
    "GATConv",
]


# Map frameworks to setup/benchmarks
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

# Benchmark of traditional loader and padded loader

# import mlx_graphs.loaders as mxg_loaders

# def fn(loader):
#     for p in loader:
#         pass


# dataset = mxg_datasets.TUDataset("BZR_MD")
# mean_num_edges = sum([g.num_edges for g in dataset]) / len(dataset)
# loaders = [
#     mxg_loaders.Dataloader(dataset, BATCH_SIZE),
#     mxg_loaders.PaddedDataloader(dataset, int(BATCH_SIZE * mean_num_edges))
# ]

# for loader in loaders:
#     times = timeit.Timer(
#         lambda: fn(loader)
#     ).repeat(repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER)

#     time = min(times) / TIMEIT_NUMBER
#     print(time)


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
    train_fn(loader, step, state, epochs=EPOCHS)


# Run benchmarks
results = [["Dataset", "Framework", "Layer", "Time/epoch"]]
for dataset_name in tqdm(DATASETS):
    print(f"Benchmarking {dataset_name}...")
    for framework in FRAMEWORKS:
        dataset = framework_to_datasets[framework](dataset_name)
        for compile in COMPILE[framework]:
            for i, layer_name in enumerate(LAYERS):
                layer = layer_classes[framework][layer_name]
                loader, step, state = framework_to_setup[framework](
                    dataset, layer, BATCH_SIZE, HIDDEN_SIZE, compile=compile
                )

                times = timeit.Timer(
                    lambda: benchmark(framework, loader, step, state)
                ).repeat(repeat=TIMEIT_REPEAT, number=TIMEIT_NUMBER)

                time = min(times) / TIMEIT_NUMBER

                results.append(
                    [
                        f"{dataset_name}",
                        f"{framework + ('(compiled)' if compile else '')}",
                        f"{layer_name}",
                        f"{time:.3f}s",
                    ]
                )

print(f"\nPlatform {platform.platform(terse=True)}")
print(f"\nmlx version: {version('mlx')}")
print(f"\nmlx-graphs version: {version('mlx_graphs')}")
print(f"\ntorch version: {version('torch')}")
print(f"\ntorch_geometric version: {version('torch_geometric')}")
print(f"\ndgl version: {version('dgl')}")
print(to_markdown_table(results))
