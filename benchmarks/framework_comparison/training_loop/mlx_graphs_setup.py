from functools import partial

import mlx.core as mx
import mlx.nn as mlx_nn
import mlx.optimizers as mlx_optim

import mlx_graphs.loaders as mxg_loaders
import mlx_graphs.nn as mxg_nn


# mlx-graphs
class MXG_model(mlx_nn.Module):
    def __init__(self, layer, in_dim, hidden_dim, out_dim, dropout=0.5):
        super(MXG_model, self).__init__()

        self.conv1 = layer(in_dim, hidden_dim)
        self.conv2 = layer(hidden_dim, hidden_dim)
        self.conv3 = layer(hidden_dim, hidden_dim)
        self.linear = mxg_nn.Linear(hidden_dim, out_dim)

        self.dropout = mlx_nn.Dropout(p=dropout)

    def __call__(self, edge_index, node_features, batch_indices, batch_size):
        h = mlx_nn.relu(self.conv1(edge_index, node_features))
        h = mlx_nn.relu(self.conv2(edge_index, h))
        h = self.conv3(edge_index, h)

        h = mxg_nn.global_mean_pool(h, batch_indices, batch_size)

        h = self.dropout(h)
        h = self.linear(h)

        return h


def loss_fn(y_hat, y, parameters=None):
    return mlx_nn.losses.cross_entropy(y_hat, y, reduction="mean")


def forward_fn(model, graph):
    y_hat = model(
        graph["edge_index"],
        graph["node_features"],
        graph["_batch_indices"],
        graph["batch_size"],
    )
    labels = graph["graph_labels"]
    loss = loss_fn(y_hat, labels, model.parameters())
    return loss, y_hat


def setup_training_mxg(dataset, layer, batch_size, hid_size, compile=True):
    # Original batch loader
    loader = mxg_loaders.Dataloader(dataset, batch_size=batch_size, shuffle=True)

    # Batch loader with padding
    # mean_num_edges = sum([g.num_edges for g in dataset]) / len(dataset)
    # batch_num_edges = int(batch_size * mean_num_edges)
    # loader = mxg_loaders.PaddedDataloader(
    #     dataset, batch_num_edges=batch_num_edges, shuffle=True
    # )

    model = MXG_model(
        layer=layer,
        in_dim=dataset.num_node_features,
        hidden_dim=hid_size,
        out_dim=dataset.num_graph_classes,
    )
    mx.eval(model.parameters())

    optimizer = mlx_optim.Adam(learning_rate=0.01)
    loss_and_grad_fn = mlx_nn.value_and_grad(model, forward_fn)

    state = [model.state, optimizer.state, mx.random.state]

    if compile:

        @partial(mx.compile, inputs=state, outputs=state)
        def step(graph):
            (loss, y_hat), grads = loss_and_grad_fn(model=model, graph=graph)
            optimizer.update(model, grads)
    else:

        def step(graph):
            (loss, y_hat), grads = loss_and_grad_fn(model=model, graph=graph)
            optimizer.update(model, grads)

    return loader, step, state


def train_mxg(loader, step, state=None, epochs=2):
    for _ in range(epochs):
        for graph in loader:
            step(graph.to_dict())
            mx.eval(state)
