import numpy as np
from pathlib import Path


def mask_hp_path(nside: int):
    return Path(__file__).parent / f"masks_hp_{nside}.npy"


def load_mask_hp(nside: int):
    return np.load(mask_hp_path(nside))
