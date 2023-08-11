import torch
from lib.datasets.sine import DataSineConfig, DataSine
from lib.datasets.spiral import DataSpiralsConfig, DataSpirals
from lib.datasets.uniform import DataUniformConfig, DataUniform
from lib.datasets.mnist import DataMNISTConfig, DataMNIST
from lib.datasets.cifar import DataCIFARConfig, DataCIFAR


class _DataFactory:
    def __init__(self):
        self.datasets = dict()
        self.datasets[DataSpiralsConfig] = DataSpirals
        self.datasets[DataSineConfig] = DataSine
        self.datasets[DataUniformConfig] = DataUniform
        self.datasets[DataMNISTConfig] = DataMNIST
        self.datasets[DataCIFARConfig] = DataCIFAR

    def register(self, config_class, data_class):
        self.datasets[config_class] = data_class

    def create(self, data_config) -> torch.utils.data.Dataset:
        return self.datasets[data_config.__class__](data_config)

    def get_class(self, data_config):
        return self.datasets[data_config.__class__]


_data_factory = None


def get_factory():
    global _data_factory
    if _data_factory is None:
        _data_factory = _DataFactory()

    return _data_factory
