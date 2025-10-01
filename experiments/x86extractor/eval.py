from pathlib import Path
import os
import sys
import importlib
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
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.serialization import deserialize_model, DeserializeConfig
from experiments.x86extractor.extract import process_elf_file
from lib.train import evaluate_metrics_on_data
from lib.render_duck import insert_checkpoint_pg, ensure_duck, attach_pg, sync


def create_config(ensemble_id, **kwargs):
    loss = torch.nn.CrossEntropyLoss()

    def cls_loss(outputs, batch):
        return loss(outputs["logits"], batch["target"])

    train_config = TrainConfig(
        model_config=MLPClassConfig(widths=[256, 256]),
        train_data_config=DataElfConfig(path="./experiments/x86extractor/elf_train"),
        val_data_config=DataElfConfig(path="./experiments/x86extractor/elf_val"),
        loss=cls_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=512,
        ensemble_id=ensemble_id,
        _version=34,
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
        validate_nth_epoch=5,
        visualize_terminal=False,
    )
    return train_run


def eval_one_config(config):
    config.epochs = 200
    config.compute_config.num_workers = 0
    deser_config = DeserializeConfig(
        train_run=config,
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config, latest_ok=True)

    ds_config = deser_config.train_run.train_config.train_data_config
    ds = data_factory.get_factory().create(ds_config)

    ensure_duck(reset=True)
    attach_pg()

    try:
        insert_checkpoint_pg(
            deser_model.model_id,
            int(config.epochs * len(ds) / config.train_config.batch_size),
            "",
        )
    except Exception as e:
        print(e)

    metrics = [
        metric() for metric in create_classification_metrics(None, 2).validation_metrics
    ]
    ds_configs = [
        deser_config.train_run.train_config.train_data_config,
        # deser_config.train_run.train_config.val_data_config,
        DataElfConfig(
            path="/proj/heal_pangu/datasets/x86/w10sys32/",
            as_seq=deser_config.train_run.train_config.train_data_config.as_seq,
            patch_size=deser_config.train_run.train_config.train_data_config.patch_size,
        ),
    ]
    for ds_config in ds_configs:
        print(ds_config)
        evaluate_metrics_on_data(
            deser_model.model,
            deser_model.model_id,
            deser_model.epoch * len(ds),
            metrics,
            ds_config,
            128,
            config.compute_config,
            device_id,
        )
    sync(config)

    # dl = torch.utils.data.DataLoader(
    #     ds,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )

    # for batch in dl:
    #     batch = {
    #         k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch.items()
    #     }
    #     output = deser_model.model(batch)
    #     print(output)
    #     break

    # import struct

    # with open("model.bin", "wb") as f:
    #     f.write(
    #         struct.pack("<II", *deser_config.train_run.train_config.model_config.widths)
    #     )
    #     f.write(struct.pack("<II", 0, 0))
    #     for name, params in deser_model.model.named_parameters():
    #         print(name)
    #         params = params.data.cpu().numpy().flatten()
    #         print(params)
    #         for p in params:
    #             f.write(struct.pack("<f", p))

    # configs = generic_ablation(create_config, dict(ensemble_id=[0]))
    # distributed_train(configs)


if __name__ == "__main__":
    device_id = ddp_setup()
    data_factory.get_factory()
    data_factory.register_dataset(DataElfConfig, DataElf)

    module_name = Path(sys.argv[1]).stem
    spec = importlib.util.spec_from_file_location(module_name, sys.argv[1])
    config_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_file)

    configs = config_file.get_all_configs()
    for config in configs:
        eval_one_config(config)
