import torch
from dataclasses import dataclass

from lib.serialize_human import serialize_human
from physicsnemo.models.graphcast import GraphCastNet

from experiments.weather.data import DataSpecHP, DataHP, DataHPConfig


@dataclass
class GraphCastPhysicsNemoConfig:
    nside: int
    mesh_level: int = 4
    processor_layers: int = 4
    hidden_dim: int = 128

    def serialize_human(self):
        return serialize_human(self.__dict__)


N_SURFACE = 4
N_UPPER_VARS = 5
N_PRESSURE_LEVELS = 13
N_TOTAL = N_SURFACE + N_UPPER_VARS * N_PRESSURE_LEVELS  # 69


class GraphCastPhysicsNemo(torch.nn.Module):
    def __init__(self, config: GraphCastPhysicsNemoConfig, data_spec: DataSpecHP):
        super().__init__()
        self.config = config
        ds = DataHP(DataHPConfig(nside=data_spec.nside, driscoll_healy=True))
        resolution = ds.dh_resolution()
        lat, lon = resolution["lat"], resolution["lon"]
        self.model = GraphCastNet(
            input_res=(lat, lon),
            input_dim_grid_nodes=N_TOTAL,
            output_dim_grid_nodes=N_TOTAL,
            mesh_level=config.mesh_level,
            processor_layers=config.processor_layers,
            hidden_dim=config.hidden_dim,
        )

    def forward(self, batch):
        x_surface = batch["input_surface"]
        x_upper = batch["input_upper"]
        # x_upper: [B, 5, 13, H, W] -> flatten to [B, 65, H, W]
        B, V, P, H, W = x_upper.shape
        x_upper_flat = x_upper.reshape(B, V * P, H, W)
        # Concat: [B, 69, H, W]
        x = torch.cat([x_surface, x_upper_flat], dim=1)
        # GraphCastNet: [1, 69, H, W] -> [1, 69, H, W]
        out = self.model(x)
        out_surface = out[:, :N_SURFACE]
        out_upper = out[:, N_SURFACE:].reshape(B, V, P, H, W)
        return dict(logits_surface=out_surface, logits_upper=out_upper)
