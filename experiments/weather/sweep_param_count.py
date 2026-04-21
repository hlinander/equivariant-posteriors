"""Sweep hyperparameters to find configs near a target parameter count.

Usage:
    uv run python experiments/weather/sweep_param_count.py
"""
from experiments.weather.models.graphcast_physicsnemo import GraphCastPhysicsNemoConfig, GraphCastPhysicsNemo
from experiments.weather.models.fengwu_physicsnemo import FengwuPhysicsNemoConfig, FengwuPhysicsNemo
from experiments.weather.data import DataSpecHP

TARGET = 11.4e6  # Pangu param count

data_spec = DataSpecHP(nside=64, n_surface=4, n_upper=5, n_pressure_levels=13)

print(f"Target: {TARGET/1e6:.1f}M (Pangu)\n")

print("=== GraphCast ===")
print(f"{'hidden':>8} {'layers':>6} {'mesh':>4} {'params':>10} {'note':>6}")
for hidden_dim in [128, 192, 256, 320, 384, 512]:
    for proc_layers in [4, 8, 12, 16]:
        for mesh_level in [4, 5, 6]:
            config = GraphCastPhysicsNemoConfig(
                nside=64, hidden_dim=hidden_dim,
                processor_layers=proc_layers, mesh_level=mesh_level,
            )
            model = GraphCastPhysicsNemo(config, data_spec)
            n = sum(p.numel() for p in model.parameters())
            note = "<--" if abs(n - TARGET) < 2e6 else ""
            if note or (proc_layers == 4 and mesh_level == 4):
                print(f"{hidden_dim:8d} {proc_layers:6d} {mesh_level:4d} {n/1e6:9.1f}M {note}")
            del model

print("\n=== FengWu ===")
print(f"{'embed_dim':>10} {'patch':>8} {'params':>10} {'note':>6}")
for embed_dim in [48, 96, 128, 192, 256, 384]:
    for patch_size in [(4, 4), (8, 8), (16, 16)]:
        config = FengwuPhysicsNemoConfig(nside=64, embed_dim=embed_dim, patch_size=patch_size)
        model = FengwuPhysicsNemo(config, data_spec)
        n = sum(p.numel() for p in model.parameters())
        note = "<--" if abs(n - TARGET) < 2e6 else ""
        if note or patch_size == (4, 4):
            print(f"{embed_dim:10d} {str(patch_size):>8} {n/1e6:9.1f}M {note}")
        del model
