import os
import torch
import pandas as pd
from pathlib import Path

from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.classification_metrics import create_classification_metrics
from lib.data_factory import DataMNISTConfig
from lib.datasets.mnist_visualization import visualize_mnist
import lib.uncertainty as uncertainty
import lib.data_factory as data_factory

import lib.model_factory as model_factory

from models import FeedforwardNeuralNetModel
from models import FeedforwardNeuralNetModel_proj
from models import FeedConfig
from models import FeedProjConfig


model_factory.get_factory().register(FeedConfig, FeedforwardNeuralNetModel)
model_factory.get_factory().register(FeedProjConfig, FeedforwardNeuralNetModel_proj)

model_kwargs = dict(
    input_dim=784,
    hidden_dim=20,
    output_dim=10,
    hidden_layers=16,
    sigma_W=1,  # initial variance of weight matrices
    sigma_b=1e-7,  # initial variance of biases
    activ_func="Tanh",  # activation function (Hardtanh, ReLU, Tanh, Linear)
)


def create_config(ensemble_id):
    train_config = TrainConfig(
        # model_config=MLPClassConfig(width=mlp_dim),
        model_config=FeedConfig(**model_kwargs),
        train_data_config=DataMNISTConfig(),
        val_data_config=DataMNISTConfig(validation=True),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.Adam, kwargs=dict(lr=0.001, weight_decay=0.001)
        ),
        batch_size=256,
        ensemble_id=ensemble_id,
    )
    train_eval = create_classification_metrics(visualize_mnist, 10)
    train_run = TrainRun(
        train_config=train_config,
        train_eval=train_eval,
        epochs=30,
        save_nth_epoch=10,
        validate_nth_epoch=5,
    )
    return train_run


if __name__ == "__main__":
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"
    ensemble_config = create_ensemble_config(create_config, 10)
    ensemble = create_ensemble(ensemble_config, device_id)

    # ds = data_factory.create(DataSpiralsConfig(seed=5, N=1000))
    ds = data_factory.get_factory().create(DataMNISTConfig(validation=True))
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        drop_last=True,
    )
    uncertainties = uncertainty.uncertainty(dataloader, ensemble, device_id)

    model_proj = FeedforwardNeuralNetModel_proj(
        FeedProjConfig(**model_kwargs), ds.data_spec()
    )
    model_proj_state = torch.load(
        Path(__file__).parent / "bottleneck_model_tanh_state.pt"
    )
    model_proj.load_state_dict(model_proj_state)
    model_proj.eval()

    projections = []
    for (xs, ys, sample_ids) in dataloader:
        model_proj(xs)
        projections.append(model_proj.bottleneck.detach().cpu().clone())

    xy = torch.concat(projections, dim=0)
    # r = torch.linalg.vector_norm(ds.uniform.xs.squeeze(), dim=-1)
    data = torch.concat(
        [
            uncertainties.MI[:, None],
            uncertainties.H[:, None],
            uncertainties.sample_ids[:, None],
            # r[:, None],
            xy,
            uncertainties.mean_pred[:, None],
        ],
        dim=-1,
    )
    df = pd.DataFrame(columns=["MI", "H", "id", "x", "y", "pred"], data=data.numpy())
    df.to_csv(Path(__file__).parent / "uncertainty_lyaponov.csv")
