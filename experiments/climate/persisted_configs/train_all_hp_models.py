import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.train_climate_baseline_multiseed import (
    create_config as create_pear_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)
from experiments.climate.persisted_configs.train_GRU_wrapped import (
    create_config as create_gru_pear_config,
    GRUTemporalWrapperConfig,
    GRUTemporalWrapper,
)
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
N_MODELS = int(os.environ.get("N_MODELS", "2"))

def create_configs():
    return (
        get_config_grid(create_pear_config, dict(
            ensemble_id=list(range(N_SEEDS)),
            climate_model_idx=list(range(N_MODELS)),
            embed_dims=[[192, 384, 384, 192],[192//2, 384//2, 384//2, 192//2]],
            batch_size=[12, 24, 48],
            #drop_rate=[0.0, 0.1],
            #lr=[1e-4, 5e-4],
        ))
    #   get_config_grid(create_gru_pear_config, dict(
    #         ensemble_id=list(range(N_SEEDS)),
    #         #climate_model_idx=list(range(N_MODELS)),
    #         #lr=[1e-4, 5e-5],
    #     ))
    )
def run(config):
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)
    #mf.register(GRUTemporalWrapperConfig, GRUTemporalWrapper)

    request_train_run(config)
    distributed_train([config])