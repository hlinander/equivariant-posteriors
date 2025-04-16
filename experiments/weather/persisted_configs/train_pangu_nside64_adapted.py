#!/usr/bin/env python
import torch
import numpy as np
from pathlib import Path

# import onnxruntime as ort

from lib.train_dataclasses import TrainConfig
from lib.train_dataclasses import TrainRun
from lib.train_dataclasses import TrainEval
from lib.train_dataclasses import OptimizerConfig
from lib.train_dataclasses import ComputeConfig
from lib.metric import create_metric


# from lib.models.healpix.swin_hp_transformer import SwinHPTransformerConfig
from experiments.weather.models.pangu import PanguParametrizedConfig

# from experiments.weather.models.swin_hp_pangu import SwinHPPangu

# from lib.models.mlp import MLPConfig
from lib.ddp import ddp_setup
from lib.ensemble import create_ensemble_config
from lib.ensemble import create_ensemble
from lib.ensemble import request_ensemble
from lib.ensemble import symlink_checkpoint_files
from lib.ensemble import is_ensemble_serialized
from lib.files import prepare_results

from lib.data_factory import get_factory as get_dataset_factory

# from lib.model_factory import get_factory as get_model_factory

from lib.render_psql import add_artifact

from lib.distributed_trainer import distributed_train

# from experiments.weather.data import DataHP
from experiments.weather.data import DataHPConfig

# from experiments.weather.metrics import anomaly_correlation_coefficient, rmse

NSIDE = 64


def create_config(ensemble_id, epoch):
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
        model_config=PanguParametrizedConfig(nside=64, embed_dim=192 // 4),
        train_data_config=DataHPConfig(nside=64, driscoll_healy=True),
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
        _version=5,
        # _version=55,
    )
    train_eval = TrainEval(
        train_metrics=[create_metric(reg_loss)],
        validation_metrics=[],
        log_gradient_norm=True,
    )  # create_regression_metrics(torch.nn.functional.l1_loss, None)
    train_run = TrainRun(
        # compute_config=ComputeConfig(distributed=True, num_workers=5, num_gpus=2),
        compute_config=ComputeConfig(),
        train_config=train_config,
        train_eval=train_eval,
        epochs=epoch,
        save_nth_epoch=1,
        validate_nth_epoch=20,
        keep_nth_epoch_checkpoints=10,
        keep_epoch_checkpoints=True,
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

    ensemble_config = create_ensemble_config(lambda eid: create_config(eid, 250), 1)
    if not is_ensemble_serialized(ensemble_config):
        request_ensemble(ensemble_config)
        distributed_train(ensemble_config.members)
        exit(0)
    # ensemble = create_ensemble(ensemble_config, device_id)
    ensemble = create_ensemble(ensemble_config, device_id)

    data_factory = get_dataset_factory()
    ds = data_factory.create(DataHPConfig(nside=NSIDE))
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    result_path = prepare_results(
        # Path(__file__).parent,
        f"{Path(__file__).stem}_{ensemble_config.members[0].train_config.model_config.__class__.__name__}_nside_{NSIDE}_lite",
        ensemble_config,
    )
    symlink_checkpoint_files(ensemble, result_path)

    # options = ort.SessionOptions()
    # options.enable_cpu_mem_arena = False
    # options.enable_mem_pattern = False
    # options.enable_mem_reuse = False
    # options.intra_op_num_threads = 16

    # cuda_provider_options = {
    # "arena_extend_strategy": "kSameAsRequested",
    # }

    # ort_session_3 = ort.InferenceSession(
    #     "experiments/weather/pangu_models/pangu_weather_3.onnx",
    #     sess_options=options,
    #     providers=[("CUDAExecutionProvider", cuda_provider_options)],
    # )
    def save_and_register(name, array):
        path = result_path / f"{name}.npy"

        np.save(
            path,
            array.detach().cpu().numpy(),
        )
        add_artifact(ensemble_config.members[0], name, path)

    # acc = anomaly_correlation_coefficient(ensemble.members[0], dl, device_id)
    # rmse = rmse(ensemble.members[0], dl, device_id)
    # breakpoint()
    for idx, batch in enumerate(dl):
        batch = {k: v.to(device_id) for k, v in batch.items()}

        #     start = time.time()
        output = ensemble.members[0](batch)
        #     model_time = time.time()
        #     print(f"Model time: {model_time - start}, Sample {batch['sample_id']}")
        save_and_register(f"{idx}_of_surface", output["logits_surface"])
        save_and_register(f"{idx}_if_surface", batch["input_surface"])
        save_and_register(f"{idx}_tf_surface", batch["target_surface"])
        save_and_register(f"{idx}_of_upper", output["logits_upper"])
        save_and_register(f"{idx}_if_upper", batch["input_upper"])
        save_and_register(f"{idx}_tf_upper", batch["target_upper"])
        if idx > 10:
            break

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
    # np.save(
    #     result_path / "pangu_pred_surface.npy",
    #     batch["input_surface"].detach().cpu().numpy()[0],
    # )
    # np.save(
    #     result_path / "pangu_pred_upper.npy",
    #     batch["input_upper"].detach().cpu().numpy()[0],
    # )
    # break
