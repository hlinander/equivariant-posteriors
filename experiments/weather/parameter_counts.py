import experiments.weather.persisted_configs.train_pangu_nside64_adapted as pangu_adapted
import experiments.weather.persisted_configs.train_nside64_single_relpos as heal_pangu
import experiments.weather.persisted_configs.train_pangu_nside64 as pangu
import lib.model_factory as mf
import lib.data_factory as df


def get_parameter_count(config):
    data_spec = (
        df.get_factory()
        .get_class(config.train_config.train_data_config)
        .data_spec(config.train_config.train_data_config)
    )
    model = mf.get_factory().create(config.train_config.model_config, data_spec)
    return sum(p.numel() for p in model.parameters())


configs = [
    pangu_adapted.create_config(0, 10),
    heal_pangu.create_config(0),
    pangu.create_config(0, 10),
]

for config in configs:
    print(config.train_config.model_config.__class__.__name__)
    n_params = get_parameter_count(config)
    print(f"{n_params:e}")
