import torch

from lib.models.dense import Dense, DenseConfig
from lib.models.conv_small import ConvSmall, ConvSmallConfig
from lib.models.conv import Conv, ConvConfig
from lib.models.conv_lap import ConvLAP, ConvLAPConfig
from lib.models.transformer import Transformer, TransformerConfig
from lib.models.mlp import MLPClass, MLPClassConfig
from lib.models.mlp import MLP, MLPConfig
from lib.models.mlp_proj import MLPProjClass, MLPProjClassConfig
from lib.models.resnet import Resnet, ResnetConfig
from lib.models.swin_transformer_v2 import SwinTiny, SwinTinyConfig

from lib.models.llama2generative import LLaMA2Generative, LLaMA2GenerativeConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPanguConfig
from experiments.weather.models.swin_hp_pangu import SwinHPPangu
from experiments.weather.models.pangu import Pangu, PanguConfig
from experiments.weather.models.pangu import PanguParametrized, PanguParametrizedConfig
from experiments.weather.models.pangu_physicsnemo import (
    PanguPhysicsNemo,
    PanguPhysicsNemoConfig,
)
from experiments.weather.models.fengwu_physicsnemo import (
    FengwuPhysicsNemo,
    FengwuPhysicsNemoConfig,
)
from experiments.weather.models.swin_hp_pangu_isolatitude import (
    SwinHPPanguIsolatitudeConfig,
)
from experiments.weather.models.swin_hp_pangu_isolatitude import SwinHPPanguIsolatitude
from experiments.weather.models.swin_hp_pangu_isolatitude_conv import (
    SwinHPPanguIsolatitudeConvConfig,
)

from experiments.weather.models.swin_hp_pangu_masks import SwinHPPanguMaskConfig
from experiments.weather.models.swin_hp_pangu_masks import SwinHPPanguMask

from experiments.weather.models.swin_hp_pangu_pad import SwinHPPanguPadConfig
from experiments.weather.models.swin_hp_pangu_pad import SwinHPPanguPad

from experiments.weather.models.swin_hp_pangu_isolatitude_conv import (
    SwinHPPanguIsolatitudeConv,
)


class _ModelFactory:
    def __init__(self):
        self.models = dict()
        self.config_classes = dict()
        self.register(DenseConfig, Dense)
        self.register(ConvSmallConfig, ConvSmall)
        self.register(ConvConfig, Conv)
        self.register(ConvLAPConfig, ConvLAP)
        self.register(TransformerConfig, Transformer)
        self.register(MLPConfig, MLP)
        self.register(MLPClassConfig, MLPClass)
        self.register(MLPProjClassConfig, MLPProjClass)
        self.register(ResnetConfig, Resnet)
        self.register(SwinTinyConfig, SwinTiny)
        self.register(LLaMA2GenerativeConfig, LLaMA2Generative)
        self.register(SwinHPPanguConfig, SwinHPPangu)
        self.register(SwinHPPanguMaskConfig, SwinHPPanguMask)
        self.register(SwinHPPanguPadConfig, SwinHPPanguPad)
        self.register(SwinHPPanguIsolatitudeConfig, SwinHPPanguIsolatitude)
        self.register(SwinHPPanguIsolatitudeConvConfig, SwinHPPanguIsolatitudeConv)
        self.register(PanguConfig, Pangu)
        self.register(PanguParametrizedConfig, PanguParametrized)
        self.register(PanguPhysicsNemoConfig, PanguPhysicsNemo)
        self.register(FengwuPhysicsNemoConfig, FengwuPhysicsNemo)

    def register(self, config_class, model_class):
        self.models[config_class.__name__] = model_class
        self.config_classes[config_class.__name__] = config_class

    def create(self, model_config, data_config) -> torch.nn.Module:
        return self.models[model_config.__class__.__name__](model_config, data_config)

    def get_class(self, model_config):
        return self.models[model_config.__class__.__name__]


_model_factory = None


def get_factory():
    global _model_factory
    if _model_factory is None:
        _model_factory = _ModelFactory()

    return _model_factory
