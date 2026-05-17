import lib.model_factory as mf
import lib.data_factory as df

from experiments.climate.data.climateset_data_hp import ClimatesetHPConfig, ClimatesetDataHP
from experiments.climate.data.climateset_data_no_hp import ClimatesetConfig, ClimatesetData
from experiments.climate.models.swin_hp_climateset import SwinHPClimatesetConfig, SwinHPClimateset
from experiments.climate.models.GRU_wrapper import GRUTemporalWrapperConfig, GRUTemporalWrapper
from experiments.climate.adapted_climateset_baselines.adapted_models.climax.climax_module import ClimaXConfig, ClimaX
from experiments.climate.adapted_climateset_baselines.adapted_models.unet import UNetConfig, UNet
from experiments.climate.adapted_climateset_baselines.adapted_models.cnn_lstm import CNNLSTMConfig, CNNLSTM_ClimateBench

import experiments.climate.persisted_configs.train_climate_pear_multiseed as swin_baseline
import experiments.climate.persisted_configs.train_GRU_wrapped as gru_wrapped
import experiments.climate.persisted_configs.train_climax_nohp as climax
import experiments.climate.persisted_configs.train_unet_nohp as unet
import experiments.climate.persisted_configs.train_cnn_lstm_nohp as cnn_lstm

df.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
df.register_dataset(ClimatesetConfig, ClimatesetData)

mf.get_factory().register(SwinHPClimatesetConfig, SwinHPClimateset)
mf.get_factory().register(GRUTemporalWrapperConfig, GRUTemporalWrapper)
mf.get_factory().register(ClimaXConfig, ClimaX)
mf.get_factory().register(UNetConfig, UNet)
mf.get_factory().register(CNNLSTMConfig, CNNLSTM_ClimateBench)


def get_parameter_count(config):
    data_spec = (
        df.get_factory()
        .get_class(config.train_config.train_data_config)
        .data_spec(config.train_config.train_data_config)
    )
    model = mf.get_factory().create(config.train_config.model_config, data_spec)
    return sum(p.numel() for p in model.parameters())


configs = [
    ("SwinHP baseline",  swin_baseline.create_config(0)),
    #("GRU-wrapped Swin", gru_wrapped.create_config(0)),
    ("ClimaX",           climax.create_config(0)),
    ("UNet",             unet.create_config(0)),
    ("CNN-LSTM",         cnn_lstm.create_config(0)),
]

for name, config in configs:
    try:
        n = get_parameter_count(config)
        print(f"{name:25s}  {n:>12,d}  ({n:.3e})")
    except Exception as e:
        print(f"{name:25s}  ERROR: {e}")

from experiments.climate.persisted_configs.train_climate_pear_multiseed import (
    create_config as create_pear_config,
)

# Unique embed_dims variants from the ablation in train_all_hp_models.py
ablation_embed_dims = [
    [192//4, 384//4, 384//4, 192//4],
    #[192//2, 384//2, 384//2, 192//2],
    #[192, 384, 384, 192],
    #[252, 504, 504, 252],
    #[192*2, 384*2, 384*2, 192*2]
    # add more here if you extend the ablation
]
ablation_depth_dims = [
    #[2, 6, 6, 2],
    [4, 12, 12, 4],
]
from itertools import product as iproduct

print("\n--- Ablation configs (parameter count varies with embed_dims and depths) ---")
for embed_dims, depths in iproduct(ablation_embed_dims, ablation_depth_dims):
    config = create_pear_config(ensemble_id=0, embed_dims=embed_dims, depths=depths)
    n = get_parameter_count(config)
    print(f"embed_dims={embed_dims}  depths={depths}  {n:>12,d}  ({n:.3e})")
