from dataclasses import dataclass
import torch

from lib.models.mlp import MLPConfig
from lib.models.mlp import MLP
from lib.dataspec import DataSpec
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.models.transformer import TransformerConfig
from lib.regression_metrics import create_regression_metrics
from lib.distributed_trainer import distributed_train
from lib.ddp import ddp_setup
from lib.generic_ablation import get_config_grid
import lib.data_factory as data_factory
from lib.data_utils import create_sample_legacy
from lib.serialize_human import serialize_human


@dataclass
class DataConfig:
    model_config: MLPConfig
    name: str = "DataConfig"
    n_eval_points: int = 10000
    size: int = 1000

    def serialize_human(self):
        return serialize_human(self.__dict__)


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
        self.model.apply(reset_model_weights)
        eval_points = torch.rand((self.config.n_eval_points, 1))
        output = self.model(dict(input=eval_points))
        params = torch.concat([p.flatten() for p in self.model.parameters()])
        return dict(
            input=torch.concat(
                [eval_points.flatten(), output["logits"].detach().flatten()]
            ),
            target=params,
            sample_id=idx,
        )

    def __len__(self):
        return self.config.size


def loss(outputs, batch, reduction="mean"):
    return torch.nn.functional.huber_loss(outputs, batch, reduction=reduction)


def create_config(width, depth, n_points):
    def loss_wrap(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPConfig(depth * [width], "relu"),
        train_data_config=DataConfig(
            model_config=MLPConfig([10, 10], "relu"), size=1000, n_eval_points=n_points
        ),
        val_data_config=DataConfig(
            model_config=MLPConfig([10, 10], "relu"), size=1000, n_eval_points=n_points
        ),
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW, kwargs=dict(weight_decay=0.0001)
        ),
        loss=loss_wrap,
        batch_size=16,
        ensemble_id=0,
        _version=1,
    )
    train_eval = create_regression_metrics(loss, None)
    train_run = TrainRun(
        project="meta",
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=500,
        save_nth_epoch=20,
        validate_nth_epoch=20,
        visualize_terminal=False,
    )
    return train_run


def create_configs():
    return get_config_grid(
        create_config,
        dict(width=[32, 64, 128, 256], depth=[2, 3], n_points=[1000, 10000, 50000]),
    )


def run(config):
    data_factory.get_factory()
    data_factory.register_dataset(DataConfig, Data)
    distributed_train([config])


if __name__ == "__main__":
    data_factory.get_factory()
    data_factory.register_dataset(DataConfig, Data)

    configs = distributed_train(get_config_grid(create_config, {}))
