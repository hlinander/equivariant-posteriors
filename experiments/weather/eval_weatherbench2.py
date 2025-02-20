from typing import Callable
import torch
from pathlib import Path
import importlib
import sys
import os
import xarray as xr

from lib.train_dataclasses import TrainRun


from lib.ddp import ddp_setup

from experiments.weather.data import DataHP
import experiments.weather.data as data
from lib.serialization import deserialize_model, DeserializeConfig


def generate_eval_zarrs(create_config: Callable[[], TrainRun], target_path):
    device_id = ddp_setup()
    train_run = create_config(0)

    deser_config = DeserializeConfig(
        train_run=train_run,
        device_id=device_id,
    )
    deser_model = deserialize_model(deser_config)
    print("Deserialized model")
    model = deser_model.model
    _ = model.eval()

    print("Dataset")
    ds = DataHP(train_run.train_config.train_data_config.validation())
    print("Dataloader")
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    for batch_idx, batch in enumerate(dl):
        print("batch")
        batch = {
            k: v.to(device_id) if hasattr(v, "to") else v for k, v in batch.items()
        }
        days = []
        for lead_day in range(1, 5):
            with torch.no_grad():
                output = model(batch)

            print("to wb2")

            batch_cpu = {
                k: v.cpu() if hasattr(v, "cpu") else v for k, v in batch.items()
            }
            output_cpu = {
                k: v.cpu() if hasattr(v, "cpu") else v for k, v in output.items()
            }
            batch["input_surface"] = output["logits_surface"]
            batch["input_upper"] = output["logits_upper"]
            # output = output.cpu()
            xds_day = data.batch_to_weatherbench2(
                batch_cpu,
                output_cpu,
                train_run.train_config.model_config.nside,
                True,
                lead_days=lead_day,
            )
            days.append(xds_day)

        xds = xr.concat(days, dim="prediction_timedelta")
        # print(xds)
        # breakpoint()
        if (target_path / "forecast.zarr").is_dir():
            xds.to_zarr(target_path / "forecast.zarr", append_dim="time")
        else:
            print("First time, writing without append..")
            xds.to_zarr(target_path / "forecast.zarr")

        # xds.to_zarr(target_path / str(batch_idx), append_dim="time")
        print(f"Saved to {target_path / str(batch_idx)}")
        # exit(0)
        # input()


def main():
    config_file = importlib.import_module(sys.argv[1])
    output_path = Path(os.environ["WEATHER"]) / "wb2_2019_fixed_5day"
    output_path.mkdir(exist_ok=True)
    open(output_path / "argv", "w").write(" ".join(sys.argv))
    generate_eval_zarrs(config_file.create_config, output_path)


if __name__ == "__main__":
    main()
