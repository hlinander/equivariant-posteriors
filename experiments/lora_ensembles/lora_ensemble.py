#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.classification_metrics import create_classification_metrics

from lib.generic_ablation import generic_ablation

import lib.data_factory as data_factory
import lib.model_factory as model_factory

from lib.files import prepare_results

from experiments.lora_ensembles.model import LLama2Config
from experiments.lora_ensembles.model import LLama2Model
from experiments.lora_ensembles.data import NLPDataset
from experiments.lora_ensembles.data import NLPDatasetConfig


LLAMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"


def create_config(ensemble_id, lora_rank):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss(outputs, batch):
        return ce_loss(outputs["logits"], batch["labels"])

    train_config = TrainConfig(
        model_config=LLama2Config(checkpoint=LLAMA_CHECKPOINT, lora_rank=lora_rank),
        train_data_config=NLPDatasetConfig(
            dataset="mehdiiraqui/twitter_disaster",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=512,
        ),
        val_data_config=NLPDatasetConfig(
            dataset="mehdiiraqui/twitter_disaster",
            model_checkpoint=LLAMA_CHECKPOINT,
            max_len=512,
            validation=True,
        ),
        loss=loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=0.001, lr=1e-4),
        ),
        batch_size=4,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=37,
    )
    train_eval = create_classification_metrics(None, 2)
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=False, num_workers=0),
        train_config=train_config,
        train_eval=train_eval,
        epochs=2,
        save_nth_epoch=1,
        validate_nth_epoch=1,
    )
    return train_run


def register_model_and_dataset():
    data_factory.get_factory()
    data_factory.register_dataset(NLPDatasetConfig, NLPDataset)

    mf = model_factory.get_factory()
    mf.register(LLama2Config, LLama2Model)


if __name__ == "__main__":
    configs = generic_ablation(
        create_config, dict(ensemble_id=[0, 1, 2], lora_rank=[8, 16, 32])
    )
    register_model_and_dataset()
    prepare_results("lora_ensemble", configs)
