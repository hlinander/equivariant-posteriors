import torch
import lib.data_factory as data_factory
from experiments.x86extractor.dataloader import DataElf, DataElfConfig

from lib.distributed_trainer import distributed_train
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.models.mlp import MLPClassConfig
from lib.generic_ablation import generic_ablation

from lib.classification_metrics import create_classification_metrics


def create_config(ensemble_id, **kwargs):
    loss = torch.nn.CrossEntropyLoss()

    def cls_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[256, 256]),
        train_data_config=DataElfConfig(path="./experiments/x86extractor/pe2"),
        val_data_config=DataElfConfig(path="./experiments/x86extractor/pe"),
        loss=cls_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=1024,
        ensemble_id=ensemble_id,
        _version=34,
    )
    train_eval = create_classification_metrics(None, 2)
    train_eval.log_gradient_norm = True
    train_run = TrainRun(
        project="x86",
        compute_config=ComputeConfig(num_workers=1),
        train_config=train_config,
        train_eval=train_eval,
        epochs=200,
        save_nth_epoch=1,
        validate_nth_epoch=5,
        visualize_terminal=False,
    )
    return train_run


if __name__ == "__main__":
    data_factory.get_factory()
    data_factory.register_dataset(DataElfConfig, DataElf)

    configs = generic_ablation(create_config, dict(ensemble_id=[0]))
    distributed_train(configs)
