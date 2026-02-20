from dataclasses import dataclass
import torch

from lib.models.mlp import MLPConfig
from lib.models.mlp import MLP
from lib.dataspec import DataSpec
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.serialization import instantiate_model
from lib.models.transformer import TransformerConfig
from lib.regression_metrics import create_regression_metric_list
from lib.distributed_trainer import distributed_train
from lib.ddp import ddp_setup
from lib.generic_ablation import get_config_grid
import lib.data_factory as data_factory
from lib.data_utils import create_sample_legacy
from lib.serialize_human import serialize_human
from lib.metric import Metric
from lib.train_dataclasses import TrainEval


@dataclass
class DataConfig:
    model_config: MLPConfig
    name: str = "DataConfig"
    n_eval_points: int = 10000
    size: int = 1000
    n_distinct_models: int = 3

    def serialize_human(self):
        return serialize_human(self.__dict__)


def load_flat_params(model, flat_params):
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].view(p.shape))
        offset += numel


def reset_model_weights(layer):
    if hasattr(layer, "reset_parameters"):
        layer.reset_parameters()
    else:
        if hasattr(layer, "children"):
            for child in layer.children():
                reset_model_weights(child)


class Data(torch.utils.data.Dataset):
    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self.model = MLP(
            data_config.model_config,
            DataSpec(torch.Size([1]), torch.Size([1]), torch.Size([1])),
        )
        torch.random.manual_seed(1)
        self.model.apply(reset_model_weights)

    @staticmethod
    def data_spec(config: DataConfig):
        model = MLP(
            config.model_config,
            DataSpec(torch.Size([1]), torch.Size([1]), torch.Size([1])),
        )
        params = torch.concat([p.flatten() for p in model.parameters()])
        return DataSpec(
            torch.Size([2 * config.n_eval_points]),
            torch.Size([params.shape[0]]),
            torch.Size([params.shape[0]]),
        )

    def __getitem__(self, idx):
        torch.random.manual_seed(1 + idx % self.config.n_distinct_models)
        self.model.apply(reset_model_weights)
        import time

        torch.random.manual_seed(time.time())
        eval_points = torch.rand((self.config.n_eval_points, 1))
        output = self.model(dict(input=eval_points))
        params = torch.concat([p.flatten() for p in self.model.parameters()])
        return dict(
            input=torch.concat(
                [eval_points.flatten(), output["logits"].detach().flatten()]
            ),
            target=params.detach(),
            sample_id=idx,
        )

    def __len__(self):
        return self.config.size


def loss(outputs, batch, reduction="mean"):
    return torch.nn.functional.huber_loss(outputs, batch, reduction=reduction)


def output_loss(output, batch, model=None, target_model=None):
    total_mse = 0
    for sample_model, sample_target_model in zip(
        output["logits"].detach(), batch["target"].detach()
    ):
        load_flat_params(model, sample_model)
        load_flat_params(target_model, sample_target_model)
        xs = torch.linspace(-0.5, 0.5, 1000)[:, None]
        out = model(dict(input=xs))["logits"]
        out_target = target_model(dict(input=xs))["logits"]
        total_mse += torch.nn.functional.mse_loss(out, out_target)
    return total_mse


def create_config(
    width, depth, n_points, target_model_width, n_distinct_models
) -> TrainRun:
    def loss_wrap(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPConfig(depth * [width], "relu"),
        train_data_config=DataConfig(
            model_config=MLPConfig([target_model_width, target_model_width], "relu"),
            size=1000,
            n_eval_points=n_points,
            n_distinct_models=n_distinct_models,
        ),
        val_data_config=DataConfig(
            model_config=MLPConfig([target_model_width, target_model_width], "relu"),
            size=1000,
            n_eval_points=n_points,
            n_distinct_models=n_distinct_models,
        ),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW, kwargs=dict(weight_decay=0.0001)
        ),
        loss=loss_wrap,
        batch_size=16,
        ensemble_id=0,
        _version=3,
    )
    model = MLP(
        train_config.train_data_config.model_config,
        DataSpec(torch.Size([1]), torch.Size([1]), torch.Size([1])),
    )
    model_target = MLP(
        train_config.train_data_config.model_config,
        DataSpec(torch.Size([1]), torch.Size([1]), torch.Size([1])),
    )
    train_eval = TrainEval(
        train_metrics=create_regression_metric_list(loss)
        + [
            lambda: Metric(
                output_loss, metric_kwargs=dict(model=model, target_model=model_target)
            )
        ],
        validation_metrics=create_regression_metric_list(loss)
        + [
            lambda: Metric(
                output_loss, metric_kwargs=dict(model=model, target_model=model_target)
            )
        ],
        data_visualizer=None,
    )
    train_run = TrainRun(
        project="meta",
        compute_config=ComputeConfig(num_workers=0, distributed=False),
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
        visualize_terminal=True,
        visualize_interval_s=2,
    )
    return train_run


def create_configs():
    return get_config_grid(
        create_config,
        dict(
            width=[256],
            depth=[2, 3, 4, 5],
            n_points=[1000],
            target_model_width=[3],
            n_distinct_models=[10, 100, 1000],
        ),
    )


def run(config):
    data_factory.get_factory()
    data_factory.register_dataset(DataConfig, Data)
    distributed_train([config])


if __name__ == "__main__":
    data_factory.get_factory()
    data_factory.register_dataset(DataConfig, Data)

    # c = create_config(256, 2, 1000, 1)
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 3)
    # ds = Data(c.train_config.train_data_config)
    # for i in range(3):
    #     d = ds[i]
    #     xs = torch.linspace(-10, 10, 100)[:, None]
    #     ys = ds.model(dict(input=xs))["logits"].detach()

    #     ax[i].plot(xs.flatten(), ys.flatten())
    # plt.show()
    configs = distributed_train(
        get_config_grid(
            create_config,
            dict(
                width=[256],
                depth=[4],
                n_points=[1000],
                target_model_width=[1],
                n_distinct_models=[1],
            ),
        )
    )
