#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataSpiralsConfig
from lib.data_factory import DataUniformConfig
import lib.data_factory as data_factory
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
from lib.lyapunov import lambda1
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.uncertainty import uncertainty


def create_config(ensemble_id):
    train_config = TrainConfig(
        model_config=MLPClassConfig(width=50),
        train_data_config=DataSpiralsConfig(seed=0, N=1000),
        val_data_config=DataSpiralsConfig(seed=1, N=500),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        batch_size=500,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_spiral, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False),
        train_config=train_config,
        train_eval=train_eval,
        epochs=1000,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()
    ensemble_config = create_ensemble_config(create_config, 5)
    ensemble = create_ensemble(ensemble_config, device_id)

    ds = data_factory.get_factory().create(DataSpiralsConfig(seed=1, N=500))
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=500,
        shuffle=False,
        drop_last=False,
    )

    dsu = data_factory.get_factory().create(DataUniformConfig(min=-1, max=1, N=100))
    dataloaderu = torch.utils.data.DataLoader(
        dsu,
        batch_size=500,
        shuffle=False,
        drop_last=False,
    )

    uq = uncertainty(dataloaderu, ensemble, device_id)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    lambdas = []
    for xs, ys, ids in dataloaderu:
        xs = xs.to(device_id)
        lambda1s = lambda1(ensemble.members[1], xs)
        lambdas.append(lambda1s)
        X = xs[:, 0, 0]
        Y = xs[:, 1, 0]
        C = lambda1s
        axs[0, 0].scatter(X, Y, c=C, s=1)

        axs[0, 1].scatter(X, Y, c=uq.A[ids])
        axs[0, 1].set_title("A")
        axs[1, 0].scatter(X, Y, c=uq.H[ids])
        axs[1, 0].set_title("H")
        axs[1, 1].scatter(X, Y, c=uq.MI[ids])
        axs[1, 1].set_title("MI")
    fig.tight_layout()
    plt.show()
    plt.savefig(Path(__file__).parent / "uq_lambda.pdf")
