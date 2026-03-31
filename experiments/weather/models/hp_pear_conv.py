import os
from pathlib import Path
from typing import Union
from typing import Sequence
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from omegaconf import DictConfig
from hydra.utils import instantiate

from hydra import compose, initialize_config_dir

from experiments.weather.data import DataHP, DataHPConfig 

def _load_original_config(config_dir: str | Path, config_name: str = "config", overrides=["model.presteps=0"]) -> DictConfig: 
    """Load Hydra config, original NVIDIA's config"""
    overrides = overrides or []
    config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

@dataclass
class HEALPixPearConvConfig:
    """
    Config for the HEALPixPearConv model. 
    This configt is used to adapt to the already existing train pipeline
    """
    model_config_path: str = "experiments/weather/persisted_configs/dlwp_healpix/configs"
    input_channels: int = 69 # 4 surface + 65 upper (13 levels * 5 variables)
    n_channels: Sequence = (136, 68, 34)
    n_layers: Sequence = (4, 2, 1)
    enable_healpixpad: bool = True

    def serialize_human(self):
        dict = self.__dict__.copy()
        return dict

class HEALPixPearConv(nn.Module):
    """
    A simple convolutional model for HEALPix data, using the PEAR structure.
    This model follows PEAR implementation: https://arxiv.org/abs/2505.17720, without the patch embeddings and
    changing attention blocks with ConvNext blocks described in this paper: https://arxiv.org/abs/2311.06253.
    """

    def __init__(self, config: HEALPixPearConvConfig, data_spec=None):
        super().__init__()

        original_config = _load_original_config(config.model_config_path)
        self.data_spec = data_spec


        self.encoder = instantiate(
            original_config['model']['encoder'],
            input_channels=config.input_channels,
            n_channels=config.n_channels,
            n_layers=config.n_layers,
            enable_healpixpad=config.enable_healpixpad,
            _convert_="all"
        )

        self.decoder = instantiate(
            original_config['model']['decoder'],
            output_channels=config.input_channels, # We want to predict 4 variables at the surface level
            n_channels=config.n_channels[::-1], # Reverse the number of channels for the decoder
            n_layers=config.n_layers[::-1], # Reverse the number of layers for the decoder
            enable_healpixpad=config.enable_healpixpad,
            _convert_="all"
        )

    def dataset_input_reshape(self, batch):
        """
        I am considering that the input shape should be:
        - (B, C, n_pix), (B, C, L, n_pix)
        """

        surface = batch['input_surface']
        upper = batch['input_upper']
        upper = upper.reshape(
            upper.shape[0],
            upper.shape[1] * upper.shape[2],
            upper.shape[3],
        )

        # Concat of surface and upper 
        x = torch.cat([surface, upper], dim=1) # (B, C, n_pix)
        x = x.reshape(
            x.shape[0],
            12,
            1, # Time dimension, we will consider it as one for now
            x.shape[1], 
            np.sqrt(x.shape[2] // 12).astype(int),
            np.sqrt(x.shape[2] // 12).astype(int),
        ) # (B, F, T, C, H, W)
        
        x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]) # (B*F*T, C, H, W)
        
        return x

    def dataset_output_reshape(self, x, batch):
        """
        x.shape is (B*F*T, C, H, W)
        I am considering that the output shape should be:
        - (B, C, n_pix), (B, C, L, n_pix)
        """
        B = batch['input_surface'].shape[0]
        F = 12
        T = 1
        C = x.shape[1]
        layer_upper = batch['input_upper'].shape[2]
        C_upper = batch['input_upper'].shape[1]
        C_surface = batch['input_surface'].shape[1]
        H = x.shape[2]
        W = x.shape[3]

        x = x.reshape(B, F, C, H, W) # (B, F, C, H, W)
        x = x.reshape(B, C, F*H*W) # (B, C, n_pix)

        x_surface = x[:, :C_surface, :] # (B, C_surface, n_pix)
        x_upper = x[:, C_surface:, :] # (B, C_upper*layer_upper, n_pix)
        x_upper = x_upper.reshape(
            x_upper.shape[0],
            C_upper,
            layer_upper,
            x_upper.shape[2],
        ) # (B, layer_upper, C_upper//layer_upper, n_pix)

        return x_surface, x_upper
    
    def find_tensors_with_grad_history(module: nn.Module):
        bad = []
        for name, sub in module.named_modules():
            for k, v in vars(sub).items():
                if torch.is_tensor(v) and v.grad_fn is not None:
                    bad.append((name, k, tuple(v.shape), type(v.grad_fn).__name__))
        return bad

    def forward(self, batch):
        """
        FOR NOW:
        batch: {
        'input_surface': (batch_size, input_time_dim, n_channels, n_pix),
        'input_upper': (batch_size, input_time_dim, n_channels, n_pix),
        'target_surface': (batch_size, output_time_dim, n_layers, n_channels, n_pix),
        'target_upper': (batch_size, output_time_dim, n_layers, n_channels, n_pix),
        }
        """

        if hasattr(self.decoder, "reset"):
            self.decoder.reset()
        
        x = self.dataset_input_reshape(batch) # (B*F*T, C, H, W)

        x = self.encoder(x)
        x = self.decoder(x) # (B*F*T, C, H, W)
        
        x_surface, x_upper = self.dataset_output_reshape(x, batch) # (B, T, C_surface, n_pix), (B, T, layer_upper, C_upper//layer_upper, n_pix)

        return dict(logits_surface=x_surface, logits_upper=x_upper)

if __name__ == "__main__":
    
    from experiments.weather.data import DataHPConvConfig, DataHPConv

    data_config = DataHPConfig(nside=64, start_year=2007, end_year=2007)
    data = DataHP(data_config) 
    keys = ['input_surface', 'input_upper']

    batch = {key: torch.asarray(data[0][key]).unsqueeze(0) for key in keys} # Add batch dimension
    print("Input shapes:", {key: value.shape for key, value in batch.items()})

    model = HEALPixPearConv(HEALPixPearConvConfig())

    pred = model(batch)

    print("Output shape:", pred['logits_surface'].shape, pred['logits_upper'].shape)



    