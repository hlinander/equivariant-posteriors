#!/usr/bin/env python
import torch
from pathlib import Path
import pandas as pd
import tqdm

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.generic_ablation import generic_ablation
from lib.distributed_trainer import distributed_train
from lib.train import load_or_create_state, do_training
from lib.stable_hash import stable_hash_small

from lib.render_psql import (
    add_artifact,
    has_artifact,
    add_parameter,
    connect_psql,
    add_metric_epoch_values,
    get_parameter,
    insert_param,
)

from lib.regression_metrics import create_regression_metrics

import lib.data_factory as data_factory

from lib.models.mlp import MLPConfig, MLP
from lib.models.transformer import TransformerConfig

from lib.lyapunov import lambda1, lambda1fast
from lib.ddp import ddp_setup
from lib.uncertainty import uncertainty
from lib.files import prepare_results
from lib.serialization import (
    deserialize_model,
    DeserializeConfig,
)
from experiments.lyapunov2.data.separated import DataSeparatedConfig
from experiments.lyapunov2.data.separated import DataSeparated
from lib.data_registry import DataSpiralsConfig

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


def ftle_filename(name):
    return f"{name}_ftle_xy.png"


def plot_r(dataframe: pd.DataFrame, output_path, name):
    dataframe.to_csv(output_path / "df.csv")
    patchwork = importr("patchwork")
    ggplot2 = importr("ggplot2")
    ggforce = importr("ggforce")
    Hmisc = importr("Hmisc")
    grid = importr("grid")
    dplyr = importr("dplyr")

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(dataframe)

    ro.globalenv["uncertainty"] = r_df

    # ro.r("uncertainty$label <- as.character(uncertainty$pred)")

    ro.r("lambda_order <- uncertainty[order(uncertainty$lambda, decreasing=TRUE),]")
    ro.r(
        """ftle_xy <- (ggplot2::ggplot(lambda_order, aes(x=x, y=y))
     + geom_raster(aes(fill=lambda), interpolate=TRUE) 
     # + geom_point(aes(color=lambda))
     + scale_fill_gradientn(colors=rainbow(3))
     + geom_circle(
            aes(x0=x0, y0=y0, r=r), 
            fill=NA,
            color="red", 
            inherit.aes = FALSE,
            data=data.frame(x0=c(0, 0), y0=c(0,0), r=c(0.5, 1.5))
            )
     )"""
    )
    path = Path(output_path) / ftle_filename(name)
    ro.r(f'ggsave("{path.as_posix()}", ftle_xy)')
    return path


def create_config(ensemble_id, layers, model_config, width):
    loss = torch.nn.functional.mse_loss
    # loss = torch.nn.functional.huber_loss

    def mse_loss(outputs, batch, reduction="mean"):
        # breakpoint()
        # x = torch.softmax(outputs["logits"], dim=-1)
        return loss(outputs["logits"], batch["target"], reduction=reduction)

    model_config_instance = model_config(layers, width)

    train_config = TrainConfig(
        # model_config=MLPClassConfig(widths=[50, 50]),
        # model_config=ResnetConfig(num_blocks=[3, 3, 3]),
        # model_config=MLPConfig(widths=[50] * layers),
        model_config=model_config_instance,
        # train_data_config=DataSeparatedConfig(),
        # val_data_config=DataSeparatedConfig(),
        train_data_config=DataSeparatedConfig(),
        val_data_config=DataSeparatedConfig(),
        loss=mse_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.SGD,
            # kwargs=dict(),
            kwargs=dict(weight_decay=0.0, lr=0.01, momentum=0.0),
            # optimizer=torch.optim.SGD,
            # kwargs=dict(weight_decay=1e-4, lr=0.05, momentum=0.9, nesterov=True),
            # kwargs=dict(weight_decay=0.0, lr=0.001),
        ),
        batch_size=64,
        ensemble_id=ensemble_id,
        _version=6,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        compute_config=ComputeConfig(num_workers=0),
        # compute_config=ComputeConfig(num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=200,
        save_nth_epoch=1,
        validate_nth_epoch=5,
        notes=dict(
            layers=f"l{layers}_w{width}_{model_config_instance.__class__.__name__}"
        ),
    )
    return train_run


if __name__ == "__main__":

    data_factory.get_factory()
    data_factory.register_dataset(DataSeparatedConfig, DataSeparated)

    configs = generic_ablation(
        create_config,
        dict(
            layers=[3],
            width=[250],
            # layers=[2],
            ensemble_id=[0],
            model_config=[
                # lambda layers, width: TransformerConfig(
                #     embed_d=width,
                #     mlp_dim=width,
                #     n_seq=2,
                #     batch_size=2**13,
                #     num_layers=layers,
                #     num_heads=1,
                #     softmax=False,
                #     activation="gelu",
                # ),
                lambda layers, width: MLPConfig(
                    widths=[width] * layers, activation="tanh"
                ),
            ],
        ),
    )
    # ensemble_config = create_ensemble_config(create_config, 1)
    distributed_train(configs)

    # ensemble = create_ensemble(ensemble_config, device_id)

    device_id = ddp_setup()

    # model = MLP(
    #     MLPConfig([200] * 2, activation="tanh"),
    #     DataSeparated.data_spec(DataSeparatedConfig()),
    # )
    # print(model)
    # exit(0)

    # for config in configs:
    #     print("Checking")
    #     state = load_or_create_state(config, device_id)
    #     do_training(config, state, device_id)

    result_path = prepare_results("separated", configs)

    dsu = data_factory.get_factory().create(DataSeparatedConfig(validation=True))
    dataloaderu = torch.utils.data.DataLoader(
        dsu,
        batch_size=500,
        shuffle=False,
        drop_last=True,
    )

    # uq = uncertainty(dataloaderu, ensemble, device_id)
    for config in configs:
        if has_artifact(config, ftle_filename(config.notes["layers"])):
            continue

        deserialized_model = deserialize_model(DeserializeConfig(config, device_id))
        if deserialized_model is None:
            continue

        model = deserialized_model.model

        if (
            config.train_config.model_config.__class__.__name__
            == TransformerConfig.__name__
        ):
            n_layers = config.train_config.model_config.num_layers
        else:
            n_layers = len(config.train_config.model_config.widths)

        df_path = result_path / f"{stable_hash_small(config)}.pickle"

        if df_path.is_file():
            df = pd.read_pickle(df_path)
        else:
            lambdas = []
            projections = []
            xy = []

            def just_logits(x):
                return model.forward_tensor(x)["logits"]

            for batch in tqdm.tqdm(dataloaderu):
                batch = {k: v.to(device_id) for k, v in batch.items()}
                xs = batch["input"]

                output = model(batch)

                lambda1s = (
                    lambda1fast(just_logits, xs.reshape(xs.shape[0], 1, *xs.shape[1:]))
                    / n_layers
                )
                lambdas.append(lambda1s.detach())
                xy.append(xs.detach())

            lambda1_tensor = torch.concat(lambdas, dim=0)
            xy_tensor = torch.concat(xy, dim=0)

            data = torch.concat(
                [
                    lambda1_tensor[:, None].cpu(),
                    xy_tensor.cpu().squeeze(),
                ],
                dim=-1,
            )
            df = pd.DataFrame(
                columns=["lambda", "x", "y"],
                data=data.numpy(),
            )
            df.to_pickle(df_path)
        file_path = plot_r(df, result_path, config.notes["layers"])
        add_artifact(config, ftle_filename(config.notes["layers"]), file_path)
        print(f"Results: {file_path}")
    # fig.tight_layout()
    # plt.show()
    # plt.savefig(Path(__file__).parent / "uq_lambda_mnist.pdf")
