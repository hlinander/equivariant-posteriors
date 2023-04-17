import torch
from dataclasses import dataclass
from lib.datasets.sine import DataSineConfig, DataSine
from lib.datasets.spiral import DataSpiralsConfig, DataSpirals


class DataFactory:
    def __init__(self):
        self.datasets = dict()
        self.datasets[DataSpiralsConfig] = DataSpirals
        self.datasets[DataSineConfig] = DataSine

    def register(self, config_class, data_class):
        self.datasets[config_class] = data_class

    def create(self, data_config) -> torch.utils.data.Dataset:
        return self.datasets[data_config.__class__](data_config)
