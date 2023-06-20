import torch
import os
import pandas as pd

from lib.data import DataFactory
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.classification_metrics import create_classification_metrics
from lib.data import DataSpiralsConfig
from lib.data import DataUniformConfig
from lib.datasets.spiral_visualization import visualize_spiral
from lib.models.mlp import MLPClassConfig
import lib.uncertainty as uncertainty


def create_config(ensemble_id):
    train_config = TrainConfig(
        model_config=MLPClassConfig(width=100),
        # model_config=TransformerConfig(
        #     embed_d=20, mlp_dim=10, n_seq=2, batch_size=500, num_layers=2, num_heads=1
        # ),
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
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
    )
    return train_run


def main():
    if torch.cuda.is_available():
        device_id = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    else:
        device_id = "cpu"
    ensemble_config = create_ensemble_config(create_config, 10)
    ensemble = create_ensemble(ensemble_config, device_id)

    data_factory = DataFactory()
    # ds = data_factory.create(DataSpiralsConfig(seed=5, N=1000))
    ds = data_factory.create(DataUniformConfig(min=-3, max=3, N=500))
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=500,
        shuffle=False,
        drop_last=True,
    )
    uncertainties = uncertainty.uncertainty(dataloader, ensemble, device_id)
    r = torch.linalg.vector_norm(ds.uniform.xs.squeeze(), dim=-1)
    data = torch.concat(
        [
            uncertainties.MI[:, None],
            uncertainties.H[:, None],
            uncertainties.sample_ids[:, None],
            r[:, None],
            ds.uniform.xs.squeeze(),
            uncertainties.mean_pred[:, None],
        ],
        dim=-1,
    )
    df = pd.DataFrame(
        columns=["MI", "H", "id", "r", "x", "y", "pred"], data=data.numpy()
    )
    df.to_csv("uncertainty_mlp.csv")


if __name__ == "__main__":
    main()
