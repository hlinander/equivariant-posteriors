import os
import sys
import numpy as np

import glob
import copy
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple, Optional

from lib.serialize_human import serialize_human
from lib.compute_env import env

import xarray as xr
import torch
import healpix

import time
import os

DATA_DIR = "/proj/heal_pangu/users/x_tagty/climateset"

@dataclass
class ClimatesetHPConfig:
    """Configuration for Climate Dataset with HEALPix projection"""
    nside: int = 32
    cache: bool = True
    normalized: bool = True

    data_dir: str = DATA_DIR
    climate_model: str = "CAS-ESM2-0"
    ensemble: str = "r3i1p1f1"
    num_ensembles: int = 1
    
    input_vars: List[str] = field(default_factory=lambda: ["BC", "CH4", "SO2", "CO2"])
    output_vars: List[str] = field(default_factory=lambda: ["tas", "pr"])
    scenarios: List[str] = field(default_factory=lambda: ["ssp585", "ssp126", "ssp370"])
    years: Union[str, List[int]] = "2015-2100"
    historical_years: Union[str, List[int], None] = "1850-2014"
    fire_type: str = "all-fires"
    
    seq_len: int = 12
    seq_to_seq: bool = True
    channels_last: bool = False 
    
    # Train/val split settings
    split: str = "train"  # "train", "val", or "all"
    val_fraction: float = 0.1
    random_seed: int = 42
    
    spatial_res: str = "250_km"
    temporal_res: str = "mon"
    
    def serialize_human(self):
        return serialize_human(self.__dict__)
    
    def _get_cache_base_dir(self):
        return Path(self.data_dir) / "cache"
    
    def input_cache_name(self):
        input_vars_str = "_".join(sorted(self.input_vars))
        scenarios_str = "_".join(sorted(self.scenarios))
        years_str = self._format_years_for_name(self.years)
        hist_str = self._format_years_for_name(self.historical_years) if "historical" in self.scenarios else "nohistoric"
        return f"inputs_nside{self.nside}_{input_vars_str}_{scenarios_str}_{years_str}_{hist_str}_{self.fire_type}"
    
    def output_cache_name(self):
        output_vars_str = "_".join(sorted(self.output_vars))
        scenarios_str = "_".join(sorted(self.scenarios))
        years_str = self._format_years_for_name(self.years)
        hist_str = self._format_years_for_name(self.historical_years) if "historical" in self.scenarios else "nohistoric"
        spacial_res_str = self.spatial_res
        return f"outputs_{self.climate_model}_{self.ensemble}_nside{self.nside}_{output_vars_str}_{scenarios_str}_{years_str}_{hist_str}_{spacial_res_str}"
    
    def _format_years_for_name(self, years):
        if isinstance(years, str):
            return years
        if isinstance(years, list):
            return f"{min(years)}-{max(years)}"
        return str(years)
    
    def get_input_cache_dir(self):
        return self._get_cache_base_dir() / self.input_cache_name()
    
    def get_output_cache_dir(self):
        return self._get_cache_base_dir() / self.output_cache_name()
    
    def get_input_stats_path(self):
        """
        Stats are tied to (cache_dir, split_seed, val_fraction) so different
        train/val configurations don't clobber each other's stats.
        Stats are always computed from the TRAIN portion only.
        """
        return self.get_input_cache_dir() / f"train_stats_seed{self.random_seed}_valfrac{self.val_fraction}.npy"
    
    def get_output_stats_path(self):
        return self.get_output_cache_dir() / f"train_stats_seed{self.random_seed}_valfrac{self.val_fraction}.npy"
    
    def validation(self):
        ret = copy.deepcopy(self)
        ret.split = "val"
        return ret
    
    def get_years_list(self, year_str: Optional[Union[str, List[int]]]) -> List[int]:
        if isinstance(year_str, list):
            return year_str
        if isinstance(year_str, int):
            return [year_str]
        splits = year_str.split("-")
        min_year, max_year = int(splits[0]), int(splits[1])
        return list(range(min_year, max_year + 1))


def deserialize_dataset_statistics(nside, config: 'ClimatesetHPConfig', data_type: str = "output"):
    if data_type == "input":
        stats_path = config.get_input_stats_path()
    else:
        stats_path = config.get_output_stats_path()
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")
    return np.load(stats_path, allow_pickle=True).item()


@dataclass
class ClimatesetDataSpec:
    nside: int
    n_input_channels: int
    n_output_channels: int
    seq_len: int


class ClimatesetDataHP(torch.utils.data.Dataset):
    """
    Climate dataset with HEALPix projection.

    Data-leakage-free normalization design
    ----------------------------------------
    The dataset always caches RAW (un-normalized) arrays.  Normalization
    statistics are computed *only* from the training split indices and saved
    to disk so that validation / test datasets can reuse exactly the same
    stats without ever touching val/test values.

    Normalization is applied lazily inside __getitem__, so the in-memory
    arrays (self.input_data / self.output_data) always stay raw.

    Typical usage
    ~~~~~~~~~~~~~
    # 1. Create train dataset – computes & saves train-split stats
    train_ds = ClimatesetDataHP(config)                 # config.split = "train"

    # 2. Create val dataset – loads stats saved by train_ds (same config)
    val_ds   = ClimatesetDataHP(config.validation())    # config.split = "val"

    # 3. Create test dataset with a different scenario
    test_cfg = copy.deepcopy(config)
    test_cfg.scenarios = ["ssp245"]
    test_cfg.split = "all"
    test_ds  = ClimatesetDataHP(test_cfg)
    # test_ds will try to load stats; pass them explicitly if the cache is new:
    test_ds.set_normalization_stats(train_ds.input_stats, train_ds.output_stats)
    """

    GAS_FOLDER_MAPPING = {
        "BC": "BC_sum",
        "CH4": "CH4_sum",
        "SO2": "SO2_sum",
        "CO2": "CO2_sum",
    }
    NO_FIRE_VARS = ["CO2"]
    
    def __init__(self, config: ClimatesetHPConfig):
        self.config = config
        
        self.input_dir  = Path(config.data_dir) / "inputs" / "input4mips"
        self.target_dir = Path(config.data_dir) / "outputs" / "CMIP6"
        
        self.input_cache_dir  = config.get_input_cache_dir()
        self.output_cache_dir = config.get_output_cache_dir()
        
        self.years            = config.get_years_list(config.years)
        self.historical_years = config.get_years_list(config.historical_years)

        # Always load/create RAW data (no normalization at this stage)
        self._load_or_create_inputs()
        self._load_or_create_outputs()

        # Apply the train/val/all split → sets self.indices
        self._apply_split()

        # -----------------------------------------------------------------
        # Normalization stats setup
        # Stats are ONLY computed from training indices.
        # Val / test datasets load stats that were saved by the train dataset.
        # -----------------------------------------------------------------
        self.input_stats  = None
        self.output_stats = None

        if config.normalized:
            self._setup_normalization_stats()
    
    # ------------------------------------------------------------------
    # Load / create helpers (raw data only)
    # ------------------------------------------------------------------

    def _load_or_create_inputs(self):
        try:
            if self.config.cache and self.input_cache_dir.exists():
                print(f"Loading cached INPUT data from {self.input_cache_dir}")
                self._load_inputs_from_cache()
            else:
                print("Creating INPUT dataset from raw files...")
                self._create_input_dataset()
                if self.config.cache:
                    print("Saving INPUT dataset to cache...")
                    self._save_inputs_to_cache()

        except (FileNotFoundError, ValueError) as e:
            print(f"\nINPUT cache error: {e}")
            print("Clearing INPUT cache and recomputing...")
            if self.input_cache_dir.exists():
                shutil.rmtree(self.input_cache_dir)
            self._create_input_dataset()
            if self.config.cache:
                self._save_inputs_to_cache()
    
    def _load_or_create_outputs(self):
        try:
            if self.config.cache and self.output_cache_dir.exists():
                print(f"Loading cached OUTPUT data from {self.output_cache_dir}")
                self._load_outputs_from_cache()
            else:
                print("Creating OUTPUT dataset from raw files...")
                self._create_output_dataset()
                if self.config.cache:
                    print("Saving OUTPUT dataset to cache...")
                    self._save_outputs_to_cache()

        except (FileNotFoundError, ValueError) as e:
            print(f"\nOUTPUT cache error: {e}")
            print("Clearing OUTPUT cache and recomputing...")
            if self.output_cache_dir.exists():
                shutil.rmtree(self.output_cache_dir)
            self._create_output_dataset()
            if self.config.cache:
                self._save_outputs_to_cache()

    # ------------------------------------------------------------------
    # Normalization stats (computed from train split, shared with val/test)
    # ------------------------------------------------------------------

    def _setup_normalization_stats(self):
        """
        If split == 'train': compute stats from train indices and save to disk.
        Otherwise: load stats from disk (must have been saved by a train dataset
        with the same config / seed / val_fraction).
        """
        input_stats_path  = self.config.get_input_stats_path()
        output_stats_path = self.config.get_output_stats_path()

        if self.config.split == "train":
            print("\nComputing normalization stats from TRAIN split only...")
            # Use _stats_indices which covers all timesteps in the selected sequences
            # (same as self.indices for seq_len=1; expanded for seq_len>1)
            stats_indices = getattr(self, "_stats_indices", self.indices)
            self.input_stats  = self._compute_stats(self.input_data,  stats_indices, self.config.input_vars,  "INPUT")
            self.output_stats = self._compute_stats(self.output_data, stats_indices, self.config.output_vars, "OUTPUT")

            # Persist so val / test datasets can reuse them
            self.input_cache_dir.mkdir(parents=True, exist_ok=True)
            self.output_cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(input_stats_path,  self.input_stats)
            np.save(output_stats_path, self.output_stats)
            print(f"  Saved INPUT  stats → {input_stats_path}")
            print(f"  Saved OUTPUT stats → {output_stats_path}")
        
        elif self.config.split in ("val", "all"):
            if input_stats_path.exists() and output_stats_path.exists():
                self.input_stats  = np.load(input_stats_path,  allow_pickle=True).item()
                self.output_stats = np.load(output_stats_path, allow_pickle=True).item()
                print(f"  Loaded INPUT  stats from {input_stats_path}")
                print(f"  Loaded OUTPUT stats from {output_stats_path}")
            elif self.config.split == "val":
                raise FileNotFoundError(
                    f"Normalization stats not found for split='val'.\n"
                    f"  Expected INPUT  stats at: {input_stats_path}\n"
                    f"  Expected OUTPUT stats at: {output_stats_path}\n"
                    f"Create the TRAIN dataset first so stats are computed and saved."
                )
            else:
                print(
                    "  WARNING: normalized=True with split='all' but no stats found on disk.\n"
                    "  Call set_normalization_stats() before using this dataset."
                )

        elif self.config.split == "test":
            # Stats must always come explicitly from the training dataset via
            # set_normalization_stats(). We never load from disk here because
            # the test config is a different scenario/cache entirely.
            print(
                "  split='test': stats must be provided explicitly.\n"
                "  Call set_normalization_stats() before using this dataset."
            )

    @staticmethod
    def _compute_stats(data: np.ndarray, indices: np.ndarray,
                       var_names: List[str], label: str) -> Dict:
        """
        Compute per-channel mean and std over (time, pixels) using only
        the provided indices (i.e. the training timesteps).

        data shape: (total_timesteps, n_channels, n_pixels)
        """
        subset = data[indices]  # (n_train, n_channels, n_pixels)
        mean = subset.mean(axis=(0, 2), keepdims=True, dtype=np.float64)  # (1, C, 1)
        std  = subset.std( axis=(0, 2), keepdims=True, dtype=np.float64)

        print(f"  {label} normalization stats (from {len(indices)} train timesteps):")
        for var, m, s in zip(var_names, mean[0, :, 0], std[0, :, 0]):
            print(f"    {var}: mean={m:.6e}, std={s:.6e}")

        return {"mean": mean, "std": std}

    def set_normalization_stats(self, input_stats: Dict, output_stats: Dict):
        """
        Manually provide normalization stats (e.g. from a train dataset object
        or loaded from disk) for a test/OOD dataset that has no train split.

        Example
        -------
        test_ds.set_normalization_stats(train_ds.input_stats, train_ds.output_stats)
        """
        if input_stats is not None:
            if "mean" not in input_stats or "std" not in input_stats:
                raise ValueError("input_stats must have 'mean' and 'std' keys")
            self.input_stats = input_stats

        if output_stats is not None:
            if "mean" not in output_stats or "std" not in output_stats:
                raise ValueError("output_stats must have 'mean' and 'std' keys")
            self.output_stats = output_stats

        print("Normalization stats set externally.")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def _apply_split(self):
        total_timesteps = len(self.input_data)
        seq_len = self.config.seq_len

        if self.config.split in ("all", "test"):
            # For seq_len > 1, only include starts where a full sequence fits
            if seq_len > 1:
                self.indices = np.arange(total_timesteps - seq_len + 1)
            else:
                self.indices = np.arange(total_timesteps)
            self._stats_indices = self.indices
            print(f"Using all {len(self.indices)} valid start indices (seq_len={seq_len})")
            return

        num_sequences   = (total_timesteps - seq_len + 1) if seq_len > 1 else total_timesteps
        sequence_starts = np.arange(0, num_sequences, seq_len)
        num_complete_sequences = len(sequence_starts)

        rng = np.random.RandomState(self.config.random_seed)
        shuffled_seq_indices = rng.permutation(num_complete_sequences)

        val_size = int(num_complete_sequences * self.config.val_fraction)

        if self.config.split == "val":
            selected_sequences = shuffled_seq_indices[:val_size]
        elif self.config.split == "train":
            selected_sequences = shuffled_seq_indices[val_size:]
        else:
            raise ValueError(f"Invalid split: '{self.config.split}'. Choose 'all', 'train', 'test' or 'val'.")

        if seq_len > 1:
            # Store only sequence start indices (one per non-overlapping sequence)
            self.indices = np.sort(sequence_starts[selected_sequences])
            # Expand to all constituent timesteps for normalization stats computation
            all_ts = []
            for s in self.indices:
                all_ts.extend(range(int(s), min(int(s) + seq_len, total_timesteps)))
            self._stats_indices = np.array(all_ts)
        else:
            self.indices = sequence_starts[selected_sequences]
            self._stats_indices = self.indices

        print(f"Split: {self.config.split}")
        print(f"  Total timesteps:    {total_timesteps}")
        print(f"  Sequence length:    {seq_len}")
        print(f"  Complete sequences: {num_complete_sequences}")
        print(f"  Using {len(selected_sequences)} sequences ({len(self.indices)} start indices)")

    # ------------------------------------------------------------------
    # __getitem__  –  normalization applied HERE, on-the-fly
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Get a single sample.

        For seq_len == 1: returns a single timestep.
            'input'  shape: (n_input_channels,  n_pixels)
            'target' shape: (n_output_channels, n_pixels)

        For seq_len > 1: returns a sequence of seq_len consecutive timesteps.
            'input'  shape: (seq_len, n_input_channels,  n_pixels)
            'target' shape: (seq_len, n_output_channels, n_pixels)
        """
        seq_len = self.config.seq_len
        start_idx = self.indices[idx]

        if seq_len > 1:
            # Sequence path: fetch seq_len consecutive timesteps
            input_sample  = self.input_data[ start_idx:start_idx + seq_len].copy()   # (T, C_in,  N)
            target_sample = self.output_data[start_idx:start_idx + seq_len].copy()   # (T, C_out, N)

            if self.config.normalized:
                if self.input_stats is None or self.output_stats is None:
                    raise RuntimeError(
                        "normalized=True but stats are not set. "
                        "Either load a train dataset first or call set_normalization_stats()."
                    )
                # mean/std shape (1, C, 1) broadcasts over (T, C, N) correctly
                input_sample  = (input_sample  - self.input_stats["mean"])  / self.input_stats["std"]
                target_sample = (target_sample - self.output_stats["mean"]) / self.output_stats["std"]

            if self.config.channels_last:
                input_sample  = np.moveaxis(input_sample,  1, -1)
                target_sample = np.moveaxis(target_sample, 1, -1)

            return {
                'input':     input_sample.astype(np.float32),
                'target':    target_sample.astype(np.float32),
                'sample_id': int(start_idx),
            }

        # Single-timestep path (seq_len == 1) — behaviour unchanged
        actual_idx    = self.indices[idx]
        input_sample  = self.input_data[actual_idx].copy()   # (C_in,  N)
        target_sample = self.output_data[actual_idx].copy()  # (C_out, N)

        if self.config.normalized:
            if self.input_stats is None or self.output_stats is None:
                raise RuntimeError(
                    "normalized=True but stats are not set. "
                    "Either load a train dataset first or call set_normalization_stats()."
                )
            input_sample  = np.squeeze((input_sample  - self.input_stats["mean"])  / self.input_stats["std"])
            target_sample = np.squeeze((target_sample - self.output_stats["mean"]) / self.output_stats["std"])

        if self.config.channels_last:
            input_sample  = np.moveaxis(input_sample,  0, -1)
            target_sample = np.moveaxis(target_sample, 0, -1)

        return {
            'input':     input_sample.astype(np.float32),
            'target':    target_sample.astype(np.float32),
            'sample_id': int(actual_idx),
        }

    # def __getitem__(self, idx):
    #     actual_idx = self.indices[idx]

    #     input_seq  = self.input_data[ actual_idx:actual_idx + self.config.seq_len].copy()
    #     target_seq = self.output_data[actual_idx:actual_idx + self.config.seq_len].copy()

    #     # Apply normalization with stored stats (stats never derived from val/test)
    #     if self.config.normalized:
    #         if self.input_stats is None or self.output_stats is None:
    #             raise RuntimeError(
    #                 "normalized=True but stats are not set. "
    #                 "Either load a train dataset first or call set_normalization_stats()."
    #             )
    #         input_seq  = (input_seq  - self.input_stats["mean"])  / self.input_stats["std"]
    #         target_seq = (target_seq - self.output_stats["mean"]) / self.output_stats["std"]

    #     if self.config.channels_last:
    #         input_seq  = np.moveaxis(input_seq,  1, -1)
    #         target_seq = np.moveaxis(target_seq, 1, -1)

    #     return {
    #         "input":     input_seq.astype(np.float32),
    #         "target":    target_seq.astype(np.float32),
    #         "sample_id": actual_idx,
    #     }
    
    def __len__(self):
        # self.indices always holds one entry per valid sample/sequence start
        return len(self.indices)

    # ------------------------------------------------------------------
    # Denormalization (operates on external arrays; raw arrays in self.*)
    # ------------------------------------------------------------------

    def denormalize_output(self, normalized_output: np.ndarray) -> np.ndarray:
        if not self.config.normalized:
            return normalized_output
        if self.output_stats is None:
            raise RuntimeError("Output stats not loaded.")
        return normalized_output * self.output_stats["std"] + self.output_stats["mean"]
    
    def denormalize_input(self, normalized_input: np.ndarray) -> np.ndarray:
        if not self.config.normalized:
            return normalized_input
        if self.input_stats is None:
            raise RuntimeError("Input stats not loaded.")
        return normalized_input * self.input_stats["std"] + self.input_stats["mean"]

    def get_normalization_stats(self) -> Dict:
        """Return stats dict compatible with set_normalization_stats()."""
        return {
            "input_stats":  self.input_stats,
            "output_stats": self.output_stats,
        }

    # ------------------------------------------------------------------
    # Raw data creation (NO normalization here)
    # ------------------------------------------------------------------

    def _create_input_dataset(self):
        """Load raw input data from netCDF files and interpolate to HEALPix."""
        all_inputs = []
        for scenario in self.config.scenarios:
            print(f"\nProcessing INPUT scenario: {scenario}")
            input_paths = self._get_input_paths(scenario)
            print("Loading and transforming input data...")
            input_hp    = self._load_and_transform_data(input_paths, coord_names=("lat", "lon"))
            print("Stacking variables...")
            input_stack = np.stack([input_hp[var] for var in self.config.input_vars], axis=1)
            print(f"  Input shape for scenario {scenario}: {input_stack.shape}")
            all_inputs.append(input_stack)

        self.input_data = np.concatenate(all_inputs, axis=0)
    
    def _create_output_dataset(self):
        """Load raw output data from netCDF files and interpolate to HEALPix."""
        all_outputs = []
        for scenario in self.config.scenarios:
            print(f"\nProcessing OUTPUT scenario: {scenario}")
            output_paths = self._get_output_paths(scenario)
            print("Loading and transforming output data...")
            output_hp    = self._load_and_transform_data(output_paths, coord_names=("y", "x"))
            print("Stacking variables...")
            output_stack = np.stack([output_hp[var] for var in self.config.output_vars], axis=1)
            print(f"  Output shape for scenario {scenario}: {output_stack.shape}")
            all_outputs.append(output_stack)

        print("\nConcatenating all OUTPUT scenarios...")
        self.output_data = np.concatenate(all_outputs, axis=0)

    # ------------------------------------------------------------------
    # Cache I/O  (raw arrays only; stats saved separately)
    # ------------------------------------------------------------------

    def _save_inputs_to_cache(self):
        print(f"\nSaving INPUT to cache: {self.input_cache_dir}")
        self.input_cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.input_cache_dir / "data.npy", self.input_data)
        config_dict = {
            "nside":            self.config.nside,
            "input_vars":       self.config.input_vars,
            "scenarios":        self.config.scenarios,
            "years":            self.config.years,
            "historical_years": self.config.historical_years,
            "fire_type":        self.config.fire_type,
            # NOTE: 'normalized' is intentionally omitted – cache is always raw.
        }
        with open(self.input_cache_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("  INPUT cache saved (raw).")
    
    def _save_outputs_to_cache(self):
        print(f"\nSaving OUTPUT to cache: {self.output_cache_dir}")
        self.output_cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.output_cache_dir / "data.npy", self.output_data)
        config_dict = {
            "nside":            self.config.nside,
            "output_vars":      self.config.output_vars,
            "scenarios":        self.config.scenarios,
            "years":            self.config.years,
            "historical_years": self.config.historical_years,
            "climate_model":    self.config.climate_model,
            "ensemble":         self.config.ensemble,
            "spatial_res":      self.config.spatial_res,
        }
        with open(self.output_cache_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("  OUTPUT cache saved (raw).")
    
    def _load_inputs_from_cache(self):
        data_file   = self.input_cache_dir / "data.npy"
        config_file = self.input_cache_dir / "config.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"INPUT cache incomplete: missing data.npy in {self.input_cache_dir}")
        
        if config_file.exists():
            with open(config_file, "r") as f:
                cached_config = json.load(f)
            if sorted(cached_config.get("input_vars", [])) != sorted(self.config.input_vars):
                raise ValueError("INPUT cache config mismatch: input_vars")
            if sorted(cached_config.get("scenarios", [])) != sorted(self.config.scenarios):
                raise ValueError("INPUT cache config mismatch: scenarios")
            # 'normalized' key no longer expected – cache is always raw

        self.input_data = np.load(data_file)
        print(f"  Loaded INPUT from cache: shape={self.input_data.shape}")
    
    def _load_outputs_from_cache(self):
        data_file   = self.output_cache_dir / "data.npy"
        config_file = self.output_cache_dir / "config.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"OUTPUT cache incomplete: missing data.npy in {self.output_cache_dir}")
        
        if config_file.exists():
            with open(config_file, "r") as f:
                cached_config = json.load(f)
            if sorted(cached_config.get("output_vars", [])) != sorted(self.config.output_vars):
                raise ValueError("OUTPUT cache config mismatch: output_vars")
            if sorted(cached_config.get("scenarios", [])) != sorted(self.config.scenarios):
                raise ValueError("OUTPUT cache config mismatch: scenarios")
            if cached_config.get("climate_model") != self.config.climate_model:
                raise ValueError("OUTPUT cache config mismatch: climate_model")
            if cached_config.get("ensemble") != self.config.ensemble:
                raise ValueError("OUTPUT cache config mismatch: ensemble")
            if cached_config.get("spatial_res") != self.config.spatial_res:
                raise ValueError("OUTPUT cache config mismatch: spatial_res")

        self.output_data = np.load(data_file)
        print(f"  Loaded OUTPUT from cache: shape={self.output_data.shape}")

    # ------------------------------------------------------------------
    # Misc helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def data_spec(config: ClimatesetHPConfig) -> ClimatesetDataSpec:
        return ClimatesetDataSpec(
            nside=config.nside,
            n_input_channels=len(config.input_vars),
            n_output_channels=len(config.output_vars),
            seq_len=config.seq_len,
        )

    def get_meta(self) -> Dict[str, Dict]:
        return {
            "input": {
                "names": self.config.input_vars,
                "units": ["kg/m2/s" if var != "CO2" else "ppm" for var in self.config.input_vars],
            },
            "output": {
                "names": self.config.output_vars,
                "units": ["K" if var == "tas" else "kg/m2/s" for var in self.config.output_vars],
            },
        }

    def get_input_statistics_path(self):
        return self.config.get_input_stats_path()

    def get_output_statistics_path(self):
        return self.config.get_output_stats_path()

    def _get_input_paths(self, scenario: str) -> Dict[str, List[str]]:
        gas_files = {gas: [] for gas in self.config.input_vars}
        preferred_years = self.historical_years if scenario == "historical" else self.years

        if preferred_years:
            years = preferred_years
        else:
            years = set()
            for gas in self.config.input_vars:
                folder_name = self.GAS_FOLDER_MAPPING[gas]
                var_dir = self.input_dir / scenario / folder_name / self.config.spatial_res / self.config.temporal_res
                years.update(self._discover_years_from_dir(var_dir))

        for gas in self.config.input_vars:
            folder_name = self.GAS_FOLDER_MAPPING[gas]
            var_dir = self.input_dir / scenario / folder_name / self.config.spatial_res / self.config.temporal_res
            for year in years:
                year_dir = var_dir / str(year)
                files = glob.glob(str(year_dir / "*.nc"))
                for f in files:
                    if gas in self.NO_FIRE_VARS or self.config.fire_type in f:
                        gas_files[gas].append(f)

        print(f"Input files: {{{', '.join([f'{k}: {len(v)}' for k, v in gas_files.items()])}}}")
        return gas_files
    
    def _get_output_paths(self, scenario: str) -> Dict[str, List[str]]:
        var_files = {var: [] for var in self.config.output_vars}
        preferred_years = self.historical_years if scenario == "historical" else self.years

        if preferred_years:
            years = preferred_years
        else:
            years = set()
            for var in self.config.output_vars:
                var_dir = (self.target_dir / self.config.climate_model /
                           self.config.ensemble / scenario / var /
                           self.config.spatial_res / self.config.temporal_res)
                years.update(self._discover_years_from_dir(var_dir))

        for var in self.config.output_vars:
            var_dir = (self.target_dir / self.config.climate_model /
                       self.config.ensemble / scenario / var /
                       self.config.spatial_res / self.config.temporal_res)
            for year in years:
                year_dir = var_dir / str(year)
                var_files[var].extend(glob.glob(str(year_dir / "*.nc")))

        print(f"Output files: {{{', '.join([f'{k}: {len(v)}' for k, v in var_files.items()])}}}")
        return var_files
    
    def _discover_years_from_dir(self, dir_path: Path) -> List[int]:
        if not dir_path.exists():
            return []
        years = []
        try:
            for p in dir_path.iterdir():
                if p.is_dir() and p.name.isdigit():
                    years.append(int(p.name))
        except Exception:
            return []
        return sorted(set(years))
    
    def _interpolate_to_hp(self, data_array: xr.DataArray,
                           coord_names: Tuple[str, str] = ("lat", "lon")) -> np.ndarray:
        print(f"Interpolating to HEALPix (shape: {data_array.shape})...", end=" ", flush=True)
        start = time.time()
        lat_name, lon_name = coord_names
        
        npix  = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(self.config.nside, np.arange(0, npix, 1), lonlat=True, nest=True)
        hlong = np.mod(hlong, 360)
        
        xlong = xr.DataArray(hlong, dims="z")
        xlat  = xr.DataArray(hlat,  dims="z")
        
        xhp = data_array.interp({lat_name: xlat, lon_name: xlong}, kwargs={"fill_value": None})
        hp_image = np.array(xhp.to_numpy(), dtype=np.float32)
        print(f"{time.time()-start:.2f}s", flush=True)
        return hp_image
    
    def _load_and_transform_data(self, file_dict: Dict[str, List[str]],
                                  coord_names: Tuple[str, str] = ("lat", "lon")) -> Dict[str, np.ndarray]:
        hp_data_dict = {}
        for var, var_files in file_dict.items():
            print(f"  Loading {var}: {len(var_files)} files", flush=True)
            t0 = time.time()
            datasets = [xr.open_dataset(f) for f in var_files]
            print(f"    Open files: {time.time()-t0:.2f}s", flush=True)
            t0 = time.time()
            ds = xr.concat(datasets, dim="time").sortby("time")
            print(f"    Concat: {time.time()-t0:.2f}s", flush=True)
            t0 = time.time()
            arr = ds.to_array().squeeze("variable")
            print(f"    To array: {time.time()-t0:.2f}s", flush=True)
            hp_data_dict[var] = self._interpolate_to_hp(arr, coord_names)
        return hp_data_dict


# ---------------------------------------------------------------------------
# Standalone utility: load train stats from disk without loading the dataset
# ---------------------------------------------------------------------------

def load_training_stats_from_config(config: ClimatesetHPConfig) -> Dict:
    """
    Load normalization statistics (computed from train split) from disk,
    without loading the full dataset.
    """
    input_stats_path  = config.get_input_stats_path()
    output_stats_path = config.get_output_stats_path()
    stats = {}

    if input_stats_path.exists():
        stats["input_stats"] = np.load(input_stats_path, allow_pickle=True).item()
        print(f"Loaded input stats from {input_stats_path}")
    else:
        print(f"WARNING: Input stats not found at {input_stats_path}")
        stats["input_stats"] = None
    
    if output_stats_path.exists():
        stats["output_stats"] = np.load(output_stats_path, allow_pickle=True).item()
        print(f"Loaded output stats from {output_stats_path}")
    else:
        print(f"WARNING: Output stats not found at {output_stats_path}")
        stats["output_stats"] = None
    
    return stats


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    config_all1 = ClimatesetHPConfig(
        nside=32,
        climate_model="CAS-ESM2-0",
        ensemble="r3i1p1f1",
        input_vars=["CH4", "SO2", "CO2", "BC"],
        output_vars=["tas", "pr"],
        scenarios=["ssp126", "ssp370", "ssp585"],
        years="2015-2100",
        seq_len=1,
        normalized=True,
        split="all",
        cache=True,
        val_fraction=0.1,
        random_seed=42,
    )

    config = ClimatesetHPConfig(
        nside=32,
        climate_model="CAS-ESM2-0",
        ensemble="r3i1p1f1",
        input_vars=["CH4", "SO2", "CO2", "BC"],
        output_vars=["tas", "pr"],
        scenarios=["ssp126", "ssp370", "ssp585"],
        years="2015-2100",
        seq_len=1,
        normalized=True,
        split="train",
        cache=True,
        val_fraction=0.1,
        random_seed=42,
    )

    print("=" * 70)
    print("Creating ALL dataset 1")
    print("=" * 70)
    all_ds1 = ClimatesetDataHP(config_all1)
    print(all_ds1[0]["input"].shape, all_ds1[0]["target"].shape)  # sanity check

    # 1. Train dataset – computes and saves stats from TRAIN indices only
    print("=" * 70)
    print("Creating TRAIN dataset")
    print("=" * 70)
    train_ds = ClimatesetDataHP(config)
    print(train_ds[0]["input"].shape, train_ds[0]["target"].shape)  # sanity check
    # 2. Val dataset – loads the stats saved above (shares raw cache)
    print("\n" + "=" * 70)
    print("Creating VAL dataset")
    print("=" * 70)
    val_ds = ClimatesetDataHP(config.validation())
    print(val_ds[0]["input"].shape, val_ds[0]["target"].shape)  # sanity check

    # 3. Second model – reuses input cache, new output cache
    print("\n" + "=" * 70)
    print("Creating second-model TRAIN dataset")
    print("=" * 70)
    model2_config = copy.deepcopy(config)
    model2_config.climate_model = "AWI-CM-1-1-MR"
    model2_config.ensemble = "r1i1p1f1"
    model2_ds = ClimatesetDataHP(model2_config)
    print(model2_ds[0]["input"].shape, model2_ds[0]["target"].shape)  # sanity check
    # 4. Out-of-distribution test set (different scenario, no inherent split)

    print("\n" + "=" * 70)
    print("Creating OOD TEST dataset")
    print("=" * 70)
    test_config = copy.deepcopy(config)
    test_config.scenarios = ["ssp245"]
    test_config.split = "test"
    test_ds = ClimatesetDataHP(test_config)
    # Test dataset gets normalization stats from the train dataset
    test_ds.set_normalization_stats(**train_ds.get_normalization_stats())
    print(test_ds[0]["input"].shape, test_ds[0]["target"].shape)  # sanity check

    # 5. Alternatively load stats from disk (no need to keep train_ds in memory)
    stats_from_disk = load_training_stats_from_config(config)
    test_ds2 = ClimatesetDataHP(test_config)
    test_ds2.set_normalization_stats(**stats_from_disk)

    # 6. Verify statistics across full datasets (not just one sample)
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        all_inputs  = np.concatenate([ds[i]["input"]  for i in range(len(ds))], axis=0)
        all_targets = np.concatenate([ds[i]["target"] for i in range(len(ds))], axis=0)
        print(f"\n{name}  (n={len(ds)})")
        print(f"  input  mean={all_inputs.mean():.4f}  std={all_inputs.std():.4f}  "
            f"range=[{all_inputs.min():.2f}, {all_inputs.max():.2f}]")
        print(f"  target mean={all_targets.mean():.4f}  std={all_targets.std():.4f}  "
            f"range=[{all_targets.min():.2f}, {all_targets.max():.2f}]")