import torch

from lib.models.dense import Dense, DenseConfig


class ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[DenseConfig] = Dense

    def register(self, config_class, model_class):
        self.models[config_class] = model_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__](model_config, data_config)
