import torch
from dataclasses import dataclass

from lib.serialize_human import serialize_human
from physicsnemo.models.fengwu import Fengwu

from experiments.weather.data import DataSpecHP, DataHP, DataHPConfig


@dataclass
class FengwuPhysicsNemoConfig:
    nside: int
    embed_dim: int = 192
    patch_size: tuple = (4, 4)

    def serialize_human(self):
        return serialize_human(self.__dict__)


# Data upper variables are ordered [z, r, t, u, v] (indices 0,1,2,3,4)
# FengWu expects [z, r, u, v, t]
# Reorder: [0, 1, 3, 4, 2]
_DATA_TO_FENGWU = [0, 1, 3, 4, 2]
_FENGWU_TO_DATA = [0, 1, 4, 2, 3]


class FengwuPhysicsNemo(torch.nn.Module):
    def __init__(self, config: FengwuPhysicsNemoConfig, data_spec: DataSpecHP):
        super().__init__()
        self.config = config
        self.n_upper = data_spec.n_upper
        ds = DataHP(DataHPConfig(nside=data_spec.nside, driscoll_healy=True))
        resolution = ds.dh_resolution()
        lat, lon = resolution["lat"], resolution["lon"]
        self.model = Fengwu(
            img_size=(lat, lon),
            pressure_level=13,
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
        )

    def forward(self, batch):
        x_surface = batch["input_surface"]
        x_upper = batch["input_upper"]
        # x_upper: [B, 5, 13, H, W] in data order [z, r, t, u, v]
        # Reorder to FengWu order [z, r, u, v, t]
        x_upper = x_upper[:, _DATA_TO_FENGWU]

        # Call encoders/fuser/decoders directly since Fengwu.forward()
        # hardcodes channel slicing for 37 pressure levels.
        m = self.model
        z, r, u, v, t = x_upper[:, 0], x_upper[:, 1], x_upper[:, 2], x_upper[:, 3], x_upper[:, 4]
        surface, skip_surface = m.encoder_surface(x_surface)
        z, skip_z = m.encoder_z(z)
        r, skip_r = m.encoder_r(r)
        u, skip_u = m.encoder_u(u)
        v, skip_v = m.encoder_v(v)
        t, skip_t = m.encoder_t(t)

        x = torch.cat(
            [e.unsqueeze(1) for e in [surface, z, r, u, v, t]], dim=1
        )
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = m.fuser(x)
        x = x.reshape(B, PL, L_SIZE, C)

        surface = m.decoder_surface(x[:, 0], skip_surface)
        z = m.decoder_z(x[:, 1], skip_z)
        r = m.decoder_r(x[:, 2], skip_r)
        u = m.decoder_u(x[:, 3], skip_u)
        v = m.decoder_v(x[:, 4], skip_v)
        t = m.decoder_t(x[:, 5], skip_t)

        # Stack upper outputs [z, r, u, v, t] -> [B, 5, 13, H, W]
        out_upper = torch.stack([z, r, u, v, t], dim=1)
        # Reorder back to data order [z, r, t, u, v]
        out_upper = out_upper[:, _FENGWU_TO_DATA]
        return dict(logits_surface=surface, logits_upper=out_upper)
