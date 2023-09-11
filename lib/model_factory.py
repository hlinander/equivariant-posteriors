import torch

from lib.models.dense import Dense, DenseConfig
from lib.models.conv_small import ConvSmall, ConvSmallConfig
from lib.models.conv import Conv, ConvConfig
from lib.models.conv_lap import ConvLAP, ConvLAPConfig
from lib.models.transformer import Transformer, TransformerConfig
from lib.models.mlp import MLPClass, MLPClassConfig
from lib.models.mlp_proj import MLPProjClass, MLPProjClassConfig
from lib.models.resnet import Resnet, ResnetConfig
from lib.models.swin_transformer_v2 import SwinTiny, SwinTinyConfig


class _ModelFactory:
    def __init__(self):
        self.models = dict()
        self.models[DenseConfig] = Dense
        self.models[ConvSmallConfig] = ConvSmall
        self.models[ConvConfig] = Conv
        self.models[ConvLAPConfig] = ConvLAP
        self.models[TransformerConfig] = Transformer
        self.models[MLPClassConfig] = MLPClass
        self.models[MLPProjClassConfig] = MLPProjClass
        self.models[ResnetConfig] = Resnet
        self.models[SwinTinyConfig] = SwinTiny

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
