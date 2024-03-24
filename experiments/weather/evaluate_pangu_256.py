#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path
from filelock import FileLock, Timeout

# import onnxruntime as ort

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric
from lib.paths import get_lock_path


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig
from experiments.weather.models.pangu import PanguConfig

# from experiments.weather.models.swin_hp_pangu import SwinHPPangu

# from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import request_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.ensemble import is_ensemble_serialized
from lib.files import prepare_results
from lib.serialization import deserialize_model, DeserializeConfig

from lib.data_factory import get_factory as get_dataset_factory

# from lib.model_factory import get_factory as get_model_factory

from lib.render_psql import (
    add_artifact,
    has_artifact,
    add_parameter,
    connect_psql,
    add_metric_epoch_values,
    get_parameter,
    insert_param,
)

from lib.distributed_trainer import distributed_train

# from experiments.weather.data import DataHP
from experiments.weather.data import DataHPConfig, Climatology, DataHP, e5_to_numpy_hp
from experiments.weather.metrics import (
    anomaly_correlation_coefficient_hp,
    rmse_hp,
    rmse_dh,
    MeteorologicalData,
    dh_numpy_to_xr_surface_hp,
)

# from experiments.weather.metrics import anomaly_correlation_coefficient, rmse

NSIDE = 256


def create_config(ensemble_id, epoch=250):
    loss = torch.nn.L1Loss()

    def reg_loss(output, batch):
        # breakpoint()
        # breakpoint()
        # return loss(output["logits_upper"], batch["target_upper"])  # + 0.25 * loss(

        return loss(output["logits_upper"], batch["target_upper"]) + 0.25 * loss(
            output["logits_surface"], batch["target_surface"]
        )
        # return loss(output["logits_surface"], batch["target_surface"])

    train_config = TrainConfig(
        extra=dict(loss_variant="full"),
        model_config=PanguConfig(nside=256),
        train_data_config=DataHPConfig(nside=256, driscoll_healy=True),
        val_data_config=None,  # DataHPConfig(nside=NSIDE),
        loss=reg_loss,
        optimizer=OptimizerConfig(
            optimizer=torch.optim.AdamW,
            kwargs=dict(weight_decay=3e-6, lr=5e-4),
            # kwargs=dict(weight_decay=3e-6, lr=5e-3),
        ),
        batch_size=1,
        ensemble_id=ensemble_id,
        # gradient_clipping=0.3,
        # _version=57,
        _version=4,
        # _version=55,
    )
    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)], validation_metrics=[]
    )  # create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        # compute_config=ComputeConfig(distributed=True, num_workers=5, num_gpus=4),
        compute_config=ComputeConfig(distributed=False, num_workers=0, num_gpus=1),
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


# def register():
# data_factory = get_dataset_factory()
# data_factory.register(DataHPConfig, DataHP)
# model_factory = get_model_factory()
# model_factory.register(SwinHPPanguConfig, SwinHPPangu)


if __name__ == "__main__":
    device_id = ddp_setup()

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        # from pickle import dump

        print("saving allocated state during OOM")
        # snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._dump_snapshot("oom_snapshot_new.pickle")
        # dump(snapshot, open("oom_snapshot.pickle", "wb"))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    # register()

    ensemble_config = create_ensemble_config(create_config, 1)
    train_run = ensemble_config.members[0]
    result_path = prepare_results(
        # Path(__file__).parent,
        f"{Path(__file__).stem}_{train_run.train_config.model_config.__class__.__name__}_nside_{NSIDE}_lite",
        ensemble_config,
    )

    def save_and_register(name, array):
        path = result_path / f"{name}.npy"

        np.save(
            path,
            array.detach().cpu().float().numpy(),
        )
        add_artifact(train_run, name, path)

    ds_rmse = DataHP(train_run.train_config.train_data_config.validation())
    dl_rmse = torch.utils.data.DataLoader(
        ds_rmse,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    ds_config_hp = train_run.train_config.train_data_config.validation().as_hp()
    ds_rmse_hp = DataHP(ds_config_hp)
    dl_rmse_hp = torch.utils.data.DataLoader(
        ds_rmse_hp,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    # torch.cuda.memory._record_memory_history(stacks="python")
    for epoch in range(1, 137, 5):
        lock = FileLock(
            get_lock_path(
                train_config=train_run.train_config, lock_name=f"eval_{epoch}"
            ),
            0.1,
        )
        try:
            lock.acquire(blocking=False)
        except Timeout:
            continue

        try:
            eval_report_version = f"eval_log.epoch.{epoch:03d}_v4"
            if get_parameter(train_run, eval_report_version) is not None:
                continue
            deser_config = DeserializeConfig(
                train_run=create_ensemble_config(
                    lambda eid: create_config(eid, epoch), 1
                ).members[0],
                device_id=device_id,
            )
            deser_model = deserialize_model(deser_config)
            if deser_model is None:
                raise Exception("No checkpoint")
            era5_meta = MeteorologicalData()

            model = deser_model.model
            model.eval()
            rmse_res = rmse_dh(model, dl_rmse, dl_rmse_hp, device_id)
            save_and_register(
                f"{deser_model.epoch:03d}_rmse_surface.npy", rmse_res.surface
            )
            save_and_register(f"{deser_model.epoch:03d}_rmse_upper.npy", rmse_res.upper)

            with connect_psql() as conn:
                for var_idx, var_data in enumerate(rmse_res.mean_surface):
                    add_metric_epoch_values(
                        conn,
                        deser_config.train_run,
                        f"rmse_surface_{era5_meta.surface.names[var_idx]}",
                        var_data.item(),
                    )
        finally:
            lock.release()

    # for idx, batch in enumerate(dl_rmse):
    #     if idx > 0:
    #         break
    #     # if has_artifact(train_run, f"debug_{idx}_surface_out.npy"):
    #     # continue
    #     batch = {k: v.to(device_id) for k, v in batch.items()}

    #     #     start = time.time()
    #     output = model(batch)
    #     e5s = dh_numpy_to_xr_surface_hp(
    #         output["logits_surface"][0].detach().cpu(),
    #         output["logits_upper"][0].detach().cpu(),
    #         ds_rmse.get_meta(),
    #     )
    #     res = e5_to_numpy_hp(e5s, 256, False)
    #     breakpoint()

    #     # output = ensemble.members[0](batch)
    #     model_time = time.time()
    #     print(f"Model time: {model_time - start}, Sample {batch['sample_id']}")
    # save_and_register(
    #     f"debug_epoch_{deser_model.epoch}_s{idx}_surface_out.npydh",
    #     output["logits_surface"],
    # )
    # save_and_register(
    #     f"debug_epoch_{deser_model.epoch}_s{idx}_upper_out.npydh",
    #     output["logits_upper"],
    # )
    # for layer_idx, (layer_name, out_slice) in enumerate(output["layer_out"]):
    # save_and_register(
    # f"debug_epoch_{deser_model.epoch}_layer_{layer_idx:02d}_{layer_name}.npydh",
    # out_slice,
    # )
    # for layer_idx, (layer_name, out_slice) in enumerate(output["layer_out"]):
    #     save_and_register(
    #         f"debug_epoch_{deser_model.epoch}_layer_{layer_idx:02d}_{layer_name}.npy",
    #         out_slice,
    #     )
    # del output