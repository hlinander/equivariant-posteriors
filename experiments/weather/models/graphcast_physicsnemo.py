import sys
import types

import torch
from dataclasses import dataclass

from lib.serialize_human import serialize_human

# graph_cast_processor.py does `import transformer_engine as te` at module level,
# but only the GraphTransformer processor actually uses it. The rest of physicsnemo
# handles missing transformer_engine via try/except. We temporarily inject a stub
# for the import, then remove it so the try/except fallbacks elsewhere work correctly.
_te_was_absent = "transformer_engine" not in sys.modules
if _te_was_absent:
    _te = types.ModuleType("transformer_engine")
    _te_pt = types.ModuleType("transformer_engine.pytorch")
    _te_pt.LayerNorm = torch.nn.LayerNorm
    _te.pytorch = _te_pt
    sys.modules["transformer_engine"] = _te
    sys.modules["transformer_engine.pytorch"] = _te_pt

from physicsnemo.models.graphcast import GraphCastNet  # noqa: E402

if _te_was_absent:
    sys.modules.pop("transformer_engine", None)
    sys.modules.pop("transformer_engine.pytorch", None)

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
