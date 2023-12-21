#!/usr/bin/env python
import torch

from experiments.lora_ensembles.generative_llm_losses import (
    generative_next_token_and_lora_l2,
    generative_next_token_loss,
)

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig

from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.models.llama2generative import LLaMA2GenerativeConfig
from lib.metric import create_metric
from lib.files import prepare_results
from lib.data_registry import DataCommonsenseQaConfig

from lib.distributed_trainer import distributed_train

from experiments.lora_ensembles.metrics import accuracy, calibration_error


# LLaMA Checkpoint
# LLaMA_CHECKPOINT = "meta-llama/Llama-2-7b-hf"
LLaMA_CHECKPOINT = "meta-llama/Llama-2-13b-hf"


# Configuration for Training
def create_config(
    ensemble_id,
    checkpoint=LLaMA_CHECKPOINT,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0.0,
    lora_l2=0.1,
    regular_l2=0,
    target_modules=["q_proj", "v_proj"],
):
    train_config = TrainConfig(
        model_config=LLaMA2GenerativeConfig(
            checkpoint=checkpoint,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_l2=lora_l2,
            target_modules=target_modules,
        ),
        train_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLaMA_CHECKPOINT,
            max_len=256,
            validation=True,
            num_samples=1,
        ),
        val_data_config=DataCommonsenseQaConfig(
            dataset="commonsense_qa",
            model_checkpoint=LLaMA_CHECKPOINT,
            max_len=256,
            validation=True,
            num_samples=1,
        ),
        loss=generative_next_token_and_lora_l2,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=regular_l2, lr=1e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        gradient_clipping=0.3,
        _version=46,
    )
    train_eval = TrainEval(
        train_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_next_token_and_lora_l2),
            create_metric(generative_next_token_loss),
            create_metric(lambda output, batch: output["lora_l2_loss"], name="lora_l2"),
        ],
        validation_metrics=[
            create_metric(accuracy),
            create_metric(calibration_error),
            create_metric(generative_next_token_and_lora_l2),
            create_metric(generative_next_token_loss),
            create_metric(lambda output, batch: output["lora_l2_loss"], name="lora_l2"),
        ],
        data_visualizer=None,
    )
    train_run = TrainRun(
        compute_config=ComputeConfig(distributed=True, num_workers=2, num_gpus=2),
        train_config=train_config,
        train_eval=train_eval,
        epochs=2,
        save_nth_epoch=1,
        validate_nth_epoch=1,
    )
    return train_run


def main():
    print("Start")
    ensemble_config = create_ensemble_config(create_config, 1)
    prepare_results("lora_ensemble", ensemble_config.members)
    print("ensemble_config finished")
    request_ensemble(ensemble_config)
    distributed_train(ensemble_config.members)
    print("ensemble finished")


if __name__ == "__main__":
    main()
