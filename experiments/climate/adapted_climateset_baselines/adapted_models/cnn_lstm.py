
import numpy as np

import time

from typing import Optional, List, Any, Dict, Tuple, Union
#from codecarbon import EmissionsTracker

#from pytorch_lightning import LightningModule
import torch

#from emulator.src.core.evaluation import evaluate_preds, evaluate_per_target_variable
#from emulator.src.utils.utils import get_loss_function, get_logger, to_DictConfig

# from emulator.src.utils.interface import reload_model_from_id
#from emulator.src.core.callbacks import PredictionPostProcessCallback
#from timm.optim import create_optimizer_v2
from lib.dataspec import DataSpec
from lib.serialize_human import serialize_human

import torch.nn as nn
from dataclasses import dataclass

@dataclass
class CNNLSTMConfig:
    num_conv_filters: int = 20
    lstm_hidden_size: int = 25
    num_lstm_layers: int = 1
    seq_to_seq: bool = True
    seq_len: int = 12
    dropout: float = 0.0
    channels_last: bool = False
    lat: int = 144
    lon: int = 96

    def serialize_human(self):
        return serialize_human(self.__dict__)


class extract_tensor(nn.Module):
    def __init__(self, seq_to_seq) -> None:
        super().__init__()
        self.seq_to_seq = seq_to_seq

    """ Helper Module to only extract output of a LSTM (ignore hidden and cell states)"""

    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        if not (self.seq_to_seq):
            tensor = tensor[:, -1, :]
        return tensor


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

class CNNLSTM_ClimateBench(nn.Module):

    # Predicts single time step only #TODO we wanna change that do we?
    # does it handle multiple or not?

    def __init__(self, config: CNNLSTMConfig, data_spec: DataSpec, **kwargs):
        super().__init__()

        self.channels_last = config.channels_last
        self.lon = config.lon
        self.lat = config.lat
        self.seq_len = config.seq_len
        self.num_input_vars = data_spec.n_input_channels
        self.num_output_vars = data_spec.n_output_channels

        seq_to_seq = config.seq_to_seq
        num_conv_filters = config.num_conv_filters
        lstm_hidden_size = config.lstm_hidden_size
        num_lstm_layers = config.num_lstm_layers
        dropout = config.dropout

        if seq_to_seq:
            self.out_seq_len = self.seq_len
        else:
            self.out_seq_len = 1

        #self.save_hyperparameters()

        self.model = torch.nn.Sequential(
            # nn.Input(shape=(slider, width, height, num_input_vars)),
            TimeDistributed(
                nn.Conv2d(
                    in_channels=self.num_input_vars,
                    out_channels=num_conv_filters,
                    kernel_size=(3, 3),
                    padding="same",
                )
            ),  # we might need to permute because not channels last ?
            nn.ReLU(),  # , input_shape=(slider, width, height, num_input_vars)),
            TimeDistributed(nn.AvgPool2d(2)),
            # TimeDistributed(nn.AdaptiveAvgPool1d(())), ##nGlobalAvgPool2d(), does not exist in pytorch
            TimeDistributed(nn.AvgPool2d((int(self.lon / 2), int(self.lat / 2)))),
            nn.Flatten(start_dim=2),
            nn.LSTM(
                input_size=num_conv_filters,
                hidden_size=lstm_hidden_size,
                num_layers=num_lstm_layers,
                batch_first=True,
            ),  # returns tuple and complete sequence
            extract_tensor(seq_to_seq),  # ignore hidden and cell state
            nn.ReLU(),
            nn.Linear(
                in_features=lstm_hidden_size,
                out_features=self.num_output_vars * self.lon * self.lat,
            ),
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, batch: dict) -> dict:
        #x = X
        X = batch["input"]

        x = X
        if self.channels_last:
            x = x.permute(
                (0, 1, 4, 2, 3)
            )  # torch con2d expects channels before height and witdth

        x = self.model(x)
        x = torch.reshape(
            x, (X.shape[0], self.out_seq_len, self.num_output_vars, self.lon, self.lat)
        )
        if self.channels_last:
            x = x.permute((0, 1, 3, 4, 2))
        x = x.nan_to_num()
        return dict(logits_output=x)

