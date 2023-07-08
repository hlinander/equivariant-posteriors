#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataMNISTConfig
from lib.datasets.mnist_visualization import visualize_mnist

# from lib.data_factory import DataUniformConfig
import lib.data_factory as data_factory
from lib.models.mlp_proj import MLPProjClassConfig
from lib.models.mlp import MLPClassConfig
from lib.lyapunov import lambda1
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.uncertainty import uncertainty


def create_config(ensemble_id):
    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[20] * 16),
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=norm_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0)
        ),
        batch_size=128,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False),
        train_config=train_config,
        train_eval=train_eval,
        epochs=30,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


def load_model(model: torch.nn.Module, train_run: TrainRun):
    state = torch.load("model.pt")
    del state["mlp_out.weight"]
    del state["mlp_out.bias"]
    model.load_state_dict(state, strict=False)
    return model


def freeze(model: torch.nn.Module, train_run: TrainRun):
    for param in model.mlp_in.parameters():
        param.requires_grad = False
    for layer in model.mlps[:-1]:
        for param in layer.parameters():
            param.requires_grad = False
    return model


def norm_loss(out, target, model):
    return torch.nn.functional.cross_entropy(out, target)


def create_config_proj(ensemble_id):
    train_config = TrainConfig(
        model_config=MLPProjClassConfig(widths=[20] * 16 + [2], store_layers=True),
        post_model_create_hook=load_model,
        model_pre_train_hook=freeze,
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        # loss=torch.nn.CrossEntropyLoss(),
        loss=norm_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(weight_decay=0.0001)
        ),
        batch_size=128,
        ensemble_id=ensemble_id,
        _version=4,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False),
        train_config=train_config,
        train_eval=train_eval,
        epochs=30,
        save_nth_epoch=1,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    ensemble_config = create_ensemble_config(create_config, 5)
    ensemble = create_ensemble(ensemble_config, device_id)

    torch.save(ensemble.members[0].state_dict(), "model.pt")

    ensemble_config_proj = create_ensemble_config(create_config_proj, 5)
    ensemble_proj = create_ensemble(ensemble_config_proj, device_id)

    # Verify frozen layer
    ps = list(ensemble.members[0].mlp_in.parameters())
    ps_proj = list(ensemble_proj.members[0].mlp_in.parameters())
    for p, p_proj in zip(ps, ps_proj):
        assert torch.equal(p, p_proj)

    dsu = data_factory.get_factory().create(DataMNISTConfig(validation=True))
    dataloaderu = torch.utils.data.DataLoader(
        dsu,
        batch_size=128,
        shuffle=False,
        drop_last=False,
    )

    uq = uncertainty(dataloaderu, ensemble, device_id)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    lambdas = []
    projections = []

    def just_logits(x):
        return ensemble.members[0](x)["logits"]

    for xs, ys, ids in dataloaderu:
        xs = xs.to(device_id)

        lambda1s = lambda1(just_logits, xs.reshape(xs.shape[0], -1))
        lambdas.append(lambda1s)
        # X = xs[:, 0, 0]
        # Y = xs[:, 1, 0]
        output = ensemble_proj.members[0](xs)
        projections.append(output["layers"][-1])
        X = output["layers"][-1][:, 0]
        Y = output["layers"][-1][:, 1]
        C = lambda1s

    lambda1_tensor = torch.concat(lambdas, dim=0)
    projection_tensor = torch.concat(projections, dim=0)

    data = torch.concat(
        [
            lambda1_tensor[:, None],
            uq.MI[:, None],
            uq.H[:, None],
            uq.sample_ids[:, None],
            projection_tensor,
            uq.mean_pred[:, None],
        ],
        dim=-1,
    )
    df = pd.DataFrame(
        columns=["lambda", "MI", "H", "id", "x", "y", "pred"], data=data.numpy()
    )
    df.to_csv(Path(__file__).parent / "uncertainty_mnist.csv")

    fig.tight_layout()
    plt.show()
    plt.savefig(Path(__file__).parent / "uq_lambda_mnist.pdf")
