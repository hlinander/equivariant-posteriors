import torch.nn as nn
from timm.models.layers import trunc_normal_

from experiments.climate.models.swin_hp_climateset import (
    SwinHPClimateset,
    SwinHPClimatesetConfig,
)


class SwinHPClimatesetFixed(SwinHPClimateset):
    """SwinHPClimateset with corrected weight initialization for Conv layers.

    PyTorch's default Kaiming init for ConvTranspose1d uses fan_in = out_channels *
    kernel_size (the transposed fan direction), which for ConvTranspose1d(96, 2, 16)
    gives fan_in=32 instead of the intended 1536, producing weights ~7x too large.
    This override applies the same trunc_normal_(std=0.02) used for Linear layers.
    """

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
