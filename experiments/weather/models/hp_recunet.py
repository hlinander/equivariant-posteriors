from pathlib import Path

from omegaconf import DictConfig
import torch
from physicsnemo.models.dlwp_healpix import HEALPixRecUNet
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from hydra.utils import instantiate
from dataclasses import dataclass
import numpy as np

@dataclass
class HEALPixRecUNetConfig:
    """
    Config for the HEALPixRecUNet model. 
    This configt is used to adapt to the already existing train pipeline
    """
    model_config_path: Path | str = Path("experiments/weather/persisted_configs/dlwp_healpix/configs")
    input_channels = 4
    output_channels = 4
    n_constants = 0
    decoder_input_channels = 1
    input_time_dim = 1
    output_time_dim = 1

    def serialize_human(self):
        dict = self.__dict__.copy()
        dict["model_config_path"] = str(dict["model_config_path"])
        return dict
    

def _load_original_config(config_dir: str | Path, config_name: str = "config", overrides=["model.presteps=0"]) -> DictConfig: 
    """Load Hydra config, original NVIDIA's config"""
    overrides = overrides or []
    config_dir = str(Path(config_dir).resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg

def _build_model_from_config(config: HEALPixRecUNetConfig) -> nn.Module:
    """Instantiate the model from the config"""

    original_cfg = _load_original_config(config.model_config_path)

    # Copy model node and inject derived values
    mcfg = OmegaConf.create(OmegaConf.to_container(original_cfg.model, resolve=True))

    mcfg["input_channels"] = config.input_channels
    mcfg["output_channels"] = config.output_channels
    mcfg["n_constants"] = config.n_constants
    mcfg["decoder_input_channels"] = config.decoder_input_channels
    mcfg['input_time_dim'] = config.input_time_dim
    mcfg['output_time_dim'] = config.output_time_dim

    return instantiate(mcfg, _convert_="all")

    
class HEALPixRecUNet(nn.Module):
    """
    Wrapper around the original HEALPixRecUNet to adapt to the already existing train pipeline, 
    which expects a model with a specific forward method and config.
    The original HEALPixRecUNet is designed to take in a list of inputs (prognostics, dec_inputs, constants),
    but the train pipeline expects a single input tensor. 
    This wrapper will adapt the input to match the expected format of the original HEALPixRecUNet.
    """
    def __init__(self, config: HEALPixRecUNetConfig, data_spec):
        super().__init__()

        self.model = _build_model_from_config(config)
        self.model.presteps = 0 # This is a hack to avoid the model trying to do any presteps, since we are handling the input adaptation in the forward method

    def __str__(self):
        return self.model.__str__()
    
    def forward(self, batch):
        """
        Adapt the input batch to match the expected format of the original HEALPixRecUNet.
        FOR NOW:
        batch: {
        'input_surface': (batch_size, input_time_dim, n_channels, n_pix),
        'input_upper': (batch_size, input_time_dim, n_channels, n_pix),
        'target_surface': (batch_size, output_time_dim, n_layers, n_channels, n_pix),
        'target_upper': (batch_size, output_time_dim, n_layers, n_channels, n_pix),
        }
        """

        prognostics = batch['input_surface'] # (batch_size, input_time_dim, n_channels, n_pix)
        prognostics = prognostics.reshape(
            prognostics.shape[0],
            12,
            prognostics.shape[1], 
            prognostics.shape[2], 
            np.sqrt(prognostics.shape[3] // 12).astype(int),
            np.sqrt(prognostics.shape[3] // 12).astype(int),
        ) # (batch_size, n_faces, input_time_dim, n_channels, height, width)

        # TODO need to decide how to hanle the decoder input and constants, for now just set to zeros
        dec_inputs = torch.zeros(
            (prognostics.shape[0],
            12, 
            prognostics.shape[2], 
            self.model.decoder_input_channels, 
            prognostics.shape[4], 
            prognostics.shape[5])
        , device=prognostics.device) # (batch_size, n_faces, t_in, decoder_input_channels, height, width)

        x_surface = self.model([prognostics, dec_inputs, None])

        B, F, T, C, H, W = x_surface.shape
        x_surface = x_surface.reshape(B, T, C, F*H*W) # (batch_size, output_time_dim, n_channels, n_pix)

        return dict(logits_surface=x_surface)


if __name__ == "__main__":
    config = HEALPixRecUNetConfig(
        model_config_path=Path("experiments/weather/persisted_configs/dlwp_healpix/configs"),
        input_channels=4,
        output_channels=4,
        n_constants=0,
        decoder_input_channels=1,
        input_time_dim=2,
        output_time_dim=1,
    )

    model = HEALPixRecUNet(config, data_spec=None)
    print(model)