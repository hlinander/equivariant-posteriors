import os
from experiments.weather.models.hp_pear_conv import HEALPixPearConv, HEALPixPearConvConfig
from experiments.weather.models.swin_hp_pangu_isolatitude import SwinHPPanguIsolatitudeConfig, SwinHPPanguIsolatitude
from experiments.weather.data import DataHPConfig, DataHP, DataHPSubset
import numpy as np
import pandas as pd
import hydra
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lib.ddp import ddp_setup

device_id = ddp_setup()

class SubModel(torch.nn.Module):
    def __init__(self, conv_model, n_layers=1):
        super(SubModel, self).__init__()
        self.conv_model = conv_model  
        self.convblock = conv_model.encoder.encoder[0][0].convblock
        self.n_layers = n_layers

    def forward(self, batch):
        x = self.conv_model.dataset_input_reshape(batch)
        for _ in range(self.n_layers):
            x = self.convblock(x)
        x_surface, x_upper = self.conv_model.dataset_output_reshape(x, batch)
        
        return dict(logits_surface=x_surface, logits_upper=x_upper)

# config_padding = HEALPixPearConvConfig(
#     n_channels = [69],
#     n_layers = [1],
#     enable_healpixpad=True
# )

# config_no_padding = HEALPixPearConvConfig(
#     n_channels = [69],
#     n_layers = [1],
#     enable_healpixpad=False
# # )

# conv_model_padding = HEALPixPearConv(config_padding)
# conv_model_no_padding = HEALPixPearConv(config_no_padding)

model = SwinHPPanguIsolatitude(
    SwinHPPanguIsolatitudeConfig(
            base_pix=12,
            nside=64,
            dev_mode=False,
            depths=[2, 6, 6, 2],
            num_heads=[6, 12, 12, 6],
            embed_dims=[192, 384, 384, 192],
            window_size=[2, 64],
            use_cos_attn=False,
            use_v2_norm_placement=True,
            drop_rate=0,
            attn_drop_rate=0,
            drop_path_rate=0,
            rel_pos_bias="none",
            shift_size=4,
            shift_strategy="ring_shift",
            ape=False,
            patch_size=16,
        ),
    )

ds = DataHP(DataHPConfig(nside=64, start_year=2019, end_year=2019))

from experiments.weather.metrics import equivariance_error


dl = DataLoader(ds, batch_size=1, shuffle=False)


df = pd.DataFrame(equivariance_error(model, dl, device=device_id, sensitivity=120, max_batches=1).surface)
plt.plot(df.iloc[0])
plt.title(f"Equivariance Error for SubModel conv layers")
plt.xlabel("Rotation Angle (degrees)")
plt.ylabel("Equivariance Error")
plt.legend()
plt.savefig("equivariance_error_submodel_isolat_layers.png")