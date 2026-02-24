import torch
from dataclasses import dataclass

from lib.serialize_human import serialize_human
from physicsnemo.models.pangu import Pangu as PhysicsNemoPangu

from experiments.weather.data import DataSpecHP, DataHP, DataHPConfig


@dataclass
class PanguPhysicsNemoConfig:
    nside: int
    embed_dim: int = 192

    def serialize_human(self):
        return serialize_human(self.__dict__)


class PanguPhysicsNemo(torch.nn.Module):
    def __init__(self, config: PanguPhysicsNemoConfig, data_spec: DataSpecHP):
        super().__init__()
        self.config = config
        ds = DataHP(DataHPConfig(nside=data_spec.nside, driscoll_healy=True))
        resolution = ds.dh_resolution()
        lat, lon = resolution["lat"], resolution["lon"]
        self.model = PhysicsNemoPangu(
            img_size=(lat, lon),
            embed_dim=config.embed_dim,
        )

    def forward(self, batch):
        x_surface = batch["input_surface"]
        x_upper = batch["input_upper"]
        surface_mask = torch.zeros(
            (3, x_surface.shape[-2], x_surface.shape[-1]),
            device=x_surface.device,
        )
        x = self.model.prepare_input(x_surface, surface_mask, x_upper)
        out_surface, out_upper = self.model(x)
        return dict(logits_surface=out_surface, logits_upper=out_upper)
