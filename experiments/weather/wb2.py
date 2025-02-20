from pathlib import Path
import os
from weatherbench2.metrics import MSE, ACC, SpatialMSE
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.evaluation import evaluate_in_memory, evaluate_with_beam
from weatherbench2 import config


obs_path = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
climatology_path = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr"
)


paths = config.Paths(
    forecast=Path(os.environ["WEATHER"]) / "wb2_2019_fixed_5day" / "forecast.zarr",
    obs=obs_path,
    output_dir=Path(os.environ["WEATHER"])
    / "wb2_2019_fixed_5day"
    / "output",  # Directory to save evaluation results
)

selection = config.Selection(
    variables=[
        "geopotential",
        "2m_temperature",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ],
    levels=[500, 700, 850],
    time_slice=slice("2019-01-01", "2019-12-31"),
)

data_config = config.Data(selection=selection, paths=paths)

eval_configs = {
    "deterministic": config.Eval(
        metrics={
            "mse": MSE(),
            "spatial_mse": SpatialMSE(),
            #          'acc': ACC(climatology=climatology)
        },
    )
}

evaluate_in_memory(data_config, eval_configs)
