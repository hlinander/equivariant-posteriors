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
from experiments.weather.data import DataHPConfig, Climatology, DataHP
from experiments.weather.metrics import (
    anomaly_correlation_coefficient_hp,
    rmse_hp,
    MeteorologicalData,
)

# from experiments.weather.metrics import anomaly_correlation_coefficient, rmse

NSIDE = 256


def create_config(ensemble_id, epoch=200):
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
        model_config=SwinHPPanguConfig(
            base_pix=12,
            nside=NSIDE,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            # num_heads=[6, 12, 12, 6],
            num_heads=[8, 16, 16, 8],
            embed_dims=[192 // 2, 384 // 2, 384 // 2, 192 // 2],
            # embed_dims=[16, 384 // 16, 384 // 16, 192 // 16],
            # embed_dims=[x for x in [16, 32, 32, 16]],
            window_size=[2, 16],  # int(32 * (NSIDE / 256)),
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,  # ,0.1,
            attn_drop_rate=0,  # ,0.1,
            drop_path_rate=0,
            rel_pos_bias="earth",
            # shift_size=8,  # int(16 * (NSIDE / 256)),
            shift_size=8,  # int(16 * (NSIDE / 256)),
            # TODO: Ring roll?
            # TODO: Compare with zero shift
            # TODO: Last layer influence? Intermediate layer artifacts?
            # TODO: Aux loss?
            # TODO: Deeper blocks might help artifacts?
            shift_strategy="nest_roll",
            ape=False,
            patch_size=16,  # int(16 * (NSIDE / 256)),
        ),
        train_data_config=DataHPConfig(nside=NSIDE),
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
        compute_config=ComputeConfig(distributed=False, num_workers=0, num_gpus=1),
        # compute_config=ComputeConfig(distributed=False, num_workers=5, num_gpus=1),
        # compute_config=ComputeConfig(distributed=True, num_workers=5, num_gpus=4),
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
    # print("Maybe training...")
    # if not is_ensemble_serialized(ensemble_config):
    #     request_ensemble(ensemble_config)
    #     distributed_train(ensemble_config.members)
    #     exit(0)
    # ensemble = create_ensemble(ensemble_config, device_id)
    # print("Creating ensemble..")
    # ensemble = create_ensemble(ensemble_config, device_id)

    # data_factory = get_dataset_factory()
    # ds = data_factory.create(DataHPConfig(nside=NSIDE))
    # dl = torch.utils.data.DataLoader(
    # ds,
    # batch_size=1,
    # shuffle=False,
    # drop_last=False,
    # )
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
    ds_acc = Climatology(train_run.train_config.train_data_config.validation())
    dl_acc = torch.utils.data.DataLoader(
        ds_acc,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    era5_meta = MeteorologicalData()

    for epoch in range(1, 137, 5):
        continue
        if epoch < 100:
            continue
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
            eval_report_version = f"eval_log.epoch.{epoch:03d}_v2"
            if get_parameter(train_run, eval_report_version) is not None:
                continue
            print(f"[eval] Epoch {epoch}")
            deser_config = DeserializeConfig(
                train_run=create_ensemble_config(
                    lambda eid: create_config(eid, epoch), 1
                ).members[0],
                device_id=device_id,
            )
            deser_model = deserialize_model(deser_config)
            if deser_model is None:
                continue
            model = deser_model.model
            model.eval()
            print("[eval] rmse")
            rmse_res = rmse_hp(model, dl_rmse, device_id)
            print("[eval] acc")
            acc = anomaly_correlation_coefficient_hp(model, dl_acc, device_id)
            save_and_register(f"{epoch:03d}_rmse_surface.npy", rmse_res.surface)
            save_and_register(f"{epoch:03d}_rmse_upper.npy", rmse_res.upper)
            save_and_register(f"{epoch:03d}_acc_surface.npy", acc.acc_unnorm_surface)
            save_and_register(f"{epoch:03d}_acc_upper.npy", acc.acc_unnorm_upper)

            with connect_psql() as conn:
                for var_idx, var_data in enumerate(rmse_res.mean_surface):
                    add_metric_epoch_values(
                        conn,
                        deser_config.train_run,
                        f"rmse_surface_{era5_meta.surface.names[var_idx]}",
                        var_data.item(),
                    )
                for var_idx, var_data in enumerate(acc.acc_surface):
                    add_metric_epoch_values(
                        conn,
                        deser_config.train_run,
                        f"acc_surface_{era5_meta.surface.names[var_idx]}",
                        var_data.item(),
                    )
                train_run_serialized = train_run.serialize_human()
                train_id = train_run_serialized["train_id"]
                ensemble_id = train_run_serialized["ensemble_id"]
                insert_param(conn, train_id, ensemble_id, eval_report_version, "done")
        finally:
            lock.release()

    # acc = anomaly_correlation_coefficient(ensemble.members[0], dl, device_id)
    # rmse = rmse(ensemble.members[0], dl, device_id)
    # breakpoint()

    # # torch.cuda.memory._record_memory_history(stacks="python")
    # for idx, batch in enumerate(dl):
    #     if idx > 1:
    #         break
    #     if has_artifact(train_run, f"{idx}_of_surface.npy"):
    #         continue
    #     batch = {k: v.to(device_id) for k, v in batch.items()}

    #     #     start = time.time()
    #     output = model(batch)
    #     # output = ensemble.members[0](batch)
    #     #     model_time = time.time()
    #     #     print(f"Model time: {model_time - start}, Sample {batch['sample_id']}")
    #     save_and_register(f"{idx}_of_surface.npy", output["logits_surface"])
    #     save_and_register(f"{idx}_if_surface.npy", batch["input_surface"])
    #     save_and_register(f"{idx}_tf_surface.npy", batch["target_surface"])
    #     save_and_register(f"{idx}_of_upper.npy", output["logits_upper"])
    #     save_and_register(f"{idx}_if_upper.npy", batch["input_upper"])
    #     save_and_register(f"{idx}_tf_upper.npy", batch["target_upper"])
    #     del output

    # ds = Climatology(
    #     ensemble_config.members[0].train_config.train_data_config.validation()
    # )
    # dl = torch.utils.data.DataLoader(
    #     ds,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    # )
    # acc = anomaly_correlation_coefficient(ensemble.members[0], dl, device_id)

    # breakpoint()
    # add_parameter(ensemble.member_configs[0], "acc_surface", acc.acc_surface)
    # add_parameter(ensemble.member_configs[0], "acc_upper", acc.acc_upper)
    # save_and_register("acc_surface.npy", acc.acc_unnorm_surface)
    # save_and_register("acc_upper.npy", acc.acc_unnorm_upper)
    # save_and_register("of_surface.npy", output["logits_surface"])
    # np.save(
    #     result_path / "if_surface.npy",
    #     batch["input_surface"].detach().cpu().numpy(),
    # )
    # np.save(
    #     result_path / "tf_surface.npy",
    #     batch["target_surface"].detach().cpu().numpy(),
    # )

    # dh, dh_target = ds.get_driscoll_healy(ids[0])
    # te5s = ds.get_template_e5s()
    # pangu_output_upper, pangu_output_surface = ort_session_3.run(
    #     None,
    #     dict(
    #         input=dh.upper.to_array().to_numpy(),
    #         input_surface=dh.surface.to_array().to_numpy(),
    #     ),
    # )
    # pangu_surface_xds = numpy_to_xds(pangu_output_surface, te5s.surface)
    # pangu_upper_xds = numpy_to_xds(pangu_output_upper, te5s.upper)
    # pangu_np_surface, pangu_np_upper = ds.e5_to_numpy(
    #     cdstest.ERA5Sample(surface=pangu_surface_xds, upper=pangu_upper_xds)
    # )
    # np.save(result_path / "pangu_pred_surface.npy", pangu_np_surface)
    # np.save(result_path / "pangu_pred_upper.npy", pangu_np_upper)
    # np.save(    #     result_path / "pangu_pred_surface.npy",
    #     batch["input_surface"].detach().cpu().numpy()[0],
    # )
    # np.save(
    #     result_path / "pangu_pred_upper.npy",
    #     batch["input_upper"].detach().cpu().numpy()[0],
    # )
    # break
