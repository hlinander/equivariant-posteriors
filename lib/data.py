import torch
from lib.datasets.sine import DataSineConfig, DataSine
from lib.datasets.spiral import DataSpiralsConfig, DataSpirals
from lib.datasets.uniform import DataUniformConfig, DataUniform


class DataFactory:
    def __init__(self):
        self.datasets = dict()
        self.datasets[DataSpiralsConfig] = DataSpirals
        self.datasets[DataSineConfig] = DataSine
        self.datasets[DataUniformConfig] = DataUniform

    def register(self, config_class, data_class):
        self.datasets[config_class] = data_class

    def create(self, data_config) -> torch.utils.data.Dataset:
        return self.datasets[data_config.__class__](data_config)

    def get_class(self, data_config):
        return self.datasets[data_config.__class__]
