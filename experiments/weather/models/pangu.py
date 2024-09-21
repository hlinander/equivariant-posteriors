import torch
from dataclasses import dataclass

from lib.serialize_human import serialize_human

# from weatherlearn.models import Pangu as WeatherLearnPangu

# from experiments.weather.WeatherLearn.weatherlearn.models.pangu import (
# Pangu as WeatherLearnPangu,
# )
from experiments.weather.weatherlearn.models.pangu.pangu import (
    Pangu as WeatherLearnPangu,
)


from experiments.weather.data import DataSpecHP, DataHP, DataHPConfig


@dataclass
class PanguConfig:
    nside: int

    def serialize_human(self):
        return serialize_human(self.__dict__)  # dict(validation=self.validation)


class Pangu(torch.nn.Module):
    def __init__(self, config: PanguConfig, data_spec: DataSpecHP):
        super().__init__()
        self.config = config
        # breakpoint()
        ds = DataHP(DataHPConfig(nside=data_spec.nside, driscoll_healy=True))
        resolution = ds.dh_resolution()
        self.model = WeatherLearnPangu(lat=resolution["lat"], lon=resolution["lon"])

    def forward(self, batch):
        x_surface = batch["input_surface"]
        x_upper = batch["input_upper"]
        surface_masks = torch.zeros(
            (3, x_surface.shape[-2], x_surface.shape[-1]),
            device=batch["input_surface"].device,
        )
        B, C, lat, lon = x_surface.shape
        out_surface, out_upper, layer_out = self.model(
            x_surface, surface_masks, x_upper
        )
        return dict(
            logits_surface=out_surface, logits_upper=out_upper, layer_out=layer_out
        )
