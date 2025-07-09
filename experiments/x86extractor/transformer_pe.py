import torch
import lib.data_factory as data_factory
from experiments.x86extractor.dataloader import DataElf, DataElfConfig

from lib.distributed_trainer import distributed_train
from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.models.mlp import MLPClassConfig
from lib.models.transformer import TransformerConfig
from lib.generic_ablation import generic_ablation

from lib.classification_metrics import create_classification_metrics

# num_heads=[4],
# embed_d=[128],
# mlp_dim=[128],
# num_layers=[1],


def create_config(
    ensemble_id, embed_d=128, mlp_dim=128, num_heads=4, num_layers=1, **kwargs
):
    loss = torch.nn.CrossEntropyLoss()

    def cls_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=TransformerConfig(
            embed_d=embed_d,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            softmax=True,
        ),
        train_data_config=DataElfConfig(
            path="/proj/heal_pangu/datasets/x86/train", as_seq=True
        ),
        val_data_config=DataElfConfig(
            path="/proj/heal_pangu/datasets/x86/val", as_seq=True
        ),
        loss=cls_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=1024,
        ensemble_id=ensemble_id,
        _version=38,
    )
    train_eval = create_classification_metrics(None, 2)
    train_eval.log_gradient_norm = True
    train_run = TrainRun(
        project="x86",
        compute_config=ComputeConfig(num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=200,
        save_nth_epoch=1,
        validate_nth_epoch=1,
        visualize_terminal=False,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=5,
    )
    return train_run


if __name__ == "__main__":
    data_factory.get_factory()
    data_factory.register_dataset(DataElfConfig, DataElf)

    # def create_config(ensemble_id, embed_d, mlp_dim, num_heads, num_layers, **kwargs):
    configs = generic_ablation(
        create_config,
        dict(
            ensemble_id=[0, 1, 2, 3, 4, 5],
            num_heads=[4],
            embed_d=[128],
            mlp_dim=[128],
            num_layers=[1],
        ),
    )
    # configs = generic_ablation(
    #     create_config,
    #     dict(
    #         ensemble_id=[0, 1, 2, 3, 4, 5],
    #         num_heads=[1, 4, 8],
    #         embed_d=[32, 64, 128],
    #         mlp_dim=[32, 64, 128],
    #         num_layers=[1, 2],
    #     ),
    # )
    distributed_train(configs)
