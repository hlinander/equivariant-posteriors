import hydra

import numpy as np

import time

from typing import Optional, List, Any, Dict, Tuple, Union
from codecarbon import EmissionsTracker

from omegaconf import DictConfig
import omegaconf
from pytorch_lightning import LightningModule
import torch

from emulator.src.core.evaluation import evaluate_preds, evaluate_per_target_variable
from emulator.src.utils.utils import get_loss_function, get_logger, to_DictConfig

# from emulator.src.utils.interface import reload_model_from_id
from emulator.src.core.callbacks import PredictionPostProcessCallback
from timm.optim import create_optimizer_v2

import torch.nn as nn


class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step"

    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*args)
        else:
            # only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(
                *[x.view(bs * seq_len, *x.shape[2:]) for x in args], **kwargs
            )
            out_shape = out.shape
            return out.view(bs, seq_len, *out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)

    def __repr__(self):
        return f"TimeDistributed({self.module})"

class UNet(nn.Module):
    """
    https://github.com/elena-orlova/SSF-project
    """

    def __init__(
        self,
        in_var_ids: List[str],
        out_var_ids: List[str],
        longitude: int = 32,
        latitude: int = 32,
        activation_function: Union[
            str, callable, None
        ] = None,  # activation after final convolution
        encoder_name="vgg11",
        datamodule_config: DictConfig = None,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        seq_len: int = 1,
        readout: str = "pooling",
        *args,
        **kwargs,
    ):
        super().__init__(datamodule_config=datamodule_config, *args, **kwargs)

        if datamodule_config is not None:
            if datamodule_config.get("channels_last") is not None:
                self.channels_last = datamodule_config.get("channels_last")
            if datamodule_config.get("lon") is not None:
                self.lon = datamodule_config.get("lon")
            if datamodule_config.get("lat") is not None:
                self.lat = datamodule_config.get("lat")
            if datamodule_config.get("seq_len") is not None:
                self.seq_len = datamodule_config.get("seq_len")
        else:
            self.lon = longitude
            self.lat = latitude
            self.channels_last = channels_last
            self.seq_len = seq_len
        self.save_hyperparameters()
        self.num_output_vars = len(out_var_ids)
        self.num_input_vars = len(in_var_ids)

        # determine padding -> lan and lot must be divisible by 32
        pad_lon = int((np.ceil(self.lon / 32) * 32) - (self.lon / 32) * 32)
        pad_lat = int((np.ceil(self.lat / 32)) * 32 - (self.lat / 32) * 32)

        self.channels_last = channels_last

        # ption 1: linear output layer
        if readout == "linear":
            self.model = torch.nn.Sequential(
                torch.nn.ConstantPad2d(
                    (pad_lat, 0, pad_lon, 0), 0
                ),  # zero padding along lon and lat
                TimeDistributed(
                    smp.Unet(
                        encoder_name=encoder_name,
                        encoder_weights=None,
                        in_channels=self.num_input_vars,
                        classes=self.num_output_vars,
                        activation=activation_function,
                    )
                ),
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=(
                        self.num_output_vars
                        * (self.lon + pad_lon)
                        * (self.lat + pad_lat)
                        * self.seq_len
                    ),
                    out_features=(
                        self.num_output_vars * self.lon * self.lat * self.seq_len
                    ),
                ),  # map back to original size
            )

        elif readout == "pooling":
            self.model = torch.nn.Sequential(
                torch.nn.ConstantPad2d(
                    (pad_lat, 0, pad_lon, 0), 0
                ),  # zero padding along lon and lat
                TimeDistributed(
                    smp.Unet(
                        encoder_name=encoder_name,
                        encoder_weights=None,
                        in_channels=self.num_input_vars,
                        classes=self.num_output_vars,
                        activation=activation_function,
                    )
                ),
                torch.nn.AdaptiveAvgPool3d(
                    output_size=(self.num_output_vars, self.lon, self.lat)
                ),  # map back to original size
            )

        else:
            self.log_text.warn(
                f"Readout {readout} is not supported. Pls choose either 'linear' or 'pooling'"
            )
            raise NotImplementedError

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)