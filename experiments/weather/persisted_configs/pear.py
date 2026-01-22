#!/usr/bin/env python
import torch

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import request_ensemble
from lib.ensemble import is_ensemble_serialized
from lib.distributed_trainer import distributed_train

from experiments.weather.models.swin_hp_pangu_pad import SwinHPPanguPadConfig
from experiments.weather.data import DataHPConfig

NSIDE = 64


def create_config(ensemble_id, epoch=200, dataset_years=10):
    loss = torch.nn.L1Loss()

    def reg_loss(output, batch):
        return loss(output["logits_upper"], batch["target_upper"]) + 0.25 * loss(
            output["logits_surface"], batch["target_surface"]
        )

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=SwinHPPanguPadConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            num_heads=[6, 12, 12, 6],
            embed_dims=[192 // 4, 384 // 4, 384 // 4, 192 // 4],
            window_size=[2, 64],
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0,
            rel_pos_bias="single",
            shift_size=4,
            shift_strategy="ring_shift",
            ape=False,
            patch_size=16,
        ),
        train_data_config=DataHPConfig(nside=NSIDE, end_year=2007 + dataset_years),
        val_data_config=None,
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        _version=17,
    )
    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)],
        validation_metrics=[],
        log_gradient_norm=True,
    )
    train_run = TrainRun(
        project="weather",
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epoch,
        save_nth_epoch=1,
        keep_epoch_checkpoints=True,
        keep_nth_epoch_checkpoints=10,
        validate_nth_epoch=20,
        visualize_terminal=False,
    )
    return train_run


if __name__ == "__main__":
    device_id = ddp_setup()

    def oom_observer(device, alloc, device_alloc, device_free):
        print("saving allocated state during OOM")
        torch.cuda.memory._dump_snapshot("oom_snapshot_new.pickle")

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    dataset_years = 10

    ensemble_config = create_ensemble_config(
        lambda eid: create_config(eid, dataset_years=dataset_years), 1
    )

    if not is_ensemble_serialized(ensemble_config):
        request_ensemble(ensemble_config)
        distributed_train(ensemble_config.members)
        exit(0)
