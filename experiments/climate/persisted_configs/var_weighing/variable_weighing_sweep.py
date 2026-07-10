import os
from lib.generic_ablation import get_config_grid

from experiments.climate.persisted_configs.var_weighing.train_climate_pear_pr_weighted_multiseed import (
    create_config as create_pear_config,
    ClimatesetHPConfig,
    ClimatesetDataHP,
    SwinHPClimatesetConfig,
    SwinHPClimateset,
)
from lib.train_distributed import request_train_run
from lib.distributed_trainer import distributed_train
import lib.data_factory as data_factory
import lib.model_factory as model_factory

N_SEEDS = int(os.environ.get("N_SEEDS", "1"))
N_MODELS = int(os.environ.get("N_MODELS", "15"))

PR_WEIGHINGS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]  # 0.5 is the default, excluded


# def create_configs():
#     return get_config_grid(create_pear_config, dict(
#         ensemble_id=list(range(N_SEEDS)),
#         climate_model_idx=list(range(N_MODELS)),
#         embed_dims=[[192*2, 384*2, 384*2, 192*2]],
#         depths=[[2, 6, 6, 2]],
#         batch_size=[12],
#         epoch=[530],
#         drop_rate=[0.0],
#         lr=[2e-4],
#         pr_variable_weighing=PR_WEIGHINGS,
#     ))

# def create_configs():
#     return get_config_grid(create_pear_config, dict(
#         ensemble_id=list(range(N_SEEDS)),
#         climate_model_idx=list(range(N_MODELS)),
#         embed_dims=[[192, 384, 384, 192]],
#         depths=[[2, 6, 6, 2]],
#         batch_size=[12],
#         epoch=[240],
#         drop_rate=[0.0],
#         lr=[2.5e-4],
#         pr_variable_weighing=PR_WEIGHINGS,
#     ))

def create_configs():
    return get_config_grid(create_pear_config, dict(
        ensemble_id=list(range(N_SEEDS)),
        climate_model_idx=list(range(N_MODELS)),
        embed_dims=[[192//4, 384//4, 384//4, 192//4]],
        depths=[[4, 12, 12, 4]],
        batch_size=[12],
        epoch=[250],
        drop_rate=[0.0],
        lr=[2e-4],
        pr_variable_weighing=PR_WEIGHINGS,
    ))

def run(config):
    data_factory.get_factory()
    data_factory.register_dataset(ClimatesetHPConfig, ClimatesetDataHP)
    mf = model_factory.get_factory()
    mf.register(SwinHPClimatesetConfig, SwinHPClimateset)

    request_train_run(config)
    distributed_train([config])
