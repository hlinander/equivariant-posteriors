import torch

from lib.models.dense import Dense, DenseConfig
from lib.models.transformer import Transformer, TransformerConfig
from lib.models.mlp import MLPClass, MLPClassConfig
from lib.models.mlp_proj import MLPProjClass, MLPProjClassConfig


class _ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[DenseConfig] = Dense
        self.models[TransformerConfig] = Transformer
        self.models[MLPClassConfig] = MLPClass
        self.models[MLPProjClassConfig] = MLPProjClass

    def register(self, config_class, model_class):
        self.models[config_class] = model_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__](model_config, data_config)

    def get_class(self, model_config):
        return self.models[model_config.__class__]


_model_factory = None


def get_factory():
    global _model_factory
    if _model_factory is None:
        _model_factory = _ModelFactory()

    return _model_factory
