import os
import sys
import numpy as np

# Test actual threading with a simple operation
import glob
import copy
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple, Optional

from lib.serialize_human import serialize_human
from lib.compute_env import env

import xarray as xr
import torch
import healpix

import time
import os

# TODO: Validation scenario should probably be specified elsewhere
VALIDATION_SCENARIOS = ["ssp126"]
VALIDATION_YEARS = "2015-2100"
DATA_DIR = "/proj/heal_pangu/users/x_tagty/climateset"

@dataclass
class ClimatesetHPConfig:
    """Configuration for Climate Dataset with HEALPix projection"""
    nside: int = 32
    cache: bool = True
    normalized: bool = True

    data_dir: str = DATA_DIR # TODO: Potentially move this to a better place
    climate_model: str = "CAS-ESM2-0" # TODO: beter solution
    ensemble: str = "r3i1p1f1" # TODO: beter solution
    num_ensembles: int = 1
    
    # Variable settings
    input_vars: List[str] = None  # ["BC", "CH4", "SO2", "CO2"]
    output_vars: List[str] = None  # ["tas", "pr"]
    scenarios: List[str] = None  # ["ssp585", "ssp126", "ssp370"]
    years: Union[str, List[int]] = "2015-2100" # TODO: Think about if I really want or need to split or think about years like this
    historical_years: Union[str, List[int], None] = "1850-2014"  # TODO: Think about if I really want or need to split or think about years like this
    fire_type: str = "all-fires"
    
    # Data processing settings
    seq_len: int = 12  # NOTE: connected to how many timesteps we get per sample, could use for level
    seq_to_seq: bool = True # NOTE: No use right now, could be future functionality
    channels_last: bool = False 
    
    # Resolution settings
    spatial_res: str = "250_km" # for file names
    temporal_res: str = "mon"  # for file names
    
    def __post_init__(self):
        if self.input_vars is None:
            self.input_vars = ["BC", "CH4", "SO2", "CO2"]
        if self.output_vars is None:
            self.output_vars = ["tas", "pr"]
        if self.scenarios is None:
            self.scenarios = ["ssp585", "ssp126", "ssp370"]
    
    def serialize_human(self):
        return serialize_human(self.__dict__)  # dict(validation=self.validation)

    def cache_name(self): # Finns i andra med, använder keys där
        """Generate cache directory name"""
        input_vars_str = "_".join(self.input_vars)
        output_vars_str = "_".join(self.output_vars)
        scenarios_str = "_".join(self.scenarios)
        return f"climate_hp_nside_{self.nside}_{self.climate_model}_{self.ensemble}_{scenarios_str}_{input_vars_str}_{output_vars_str}"
    
    def statistics_name(self): # Finns i andra med, använder keys där 
        """Generate statistics file name"""
        return f"{self.cache_name()}_stats"
    
    def validation(self):
        ret = copy.deepcopy(self)
        ret.scenarios = VALIDATION_SCENARIOS
        ret.years = VALIDATION_YEARS
        ret.historical_years = None
        return ret
    
    def get_years_list(self, year_str: Union[str, List[int]]) -> List[int]:
        """Convert year string '2015-2100' to list of years"""
        if isinstance(year_str, list):
            return year_str
        if isinstance(year_str, int):
            return [year_str]
        
        splits = year_str.split("-")
        min_year, max_year = int(splits[0]), int(splits[1])
        # print(f"Years from {min_year} to {max_year}")
        # print(f"Here is split: {list(range(min_year, max_year + 1))}")
        return list(range(min_year, max_year + 1))

def deserialize_dataset_statistics(nside):
    ds = ClimatesetDataHP(ClimatesetHPConfig(nside=nside))
    return np.load(ds.get_statistics_path(), allow_pickle=True)


@dataclass
class ClimatesetDataSpec:
    """Data specification for climate dataset"""
    nside: int
    n_input_channels: int  # Number of input gas channels
    n_output_channels: int  # Number of output climate variables
    seq_len: int


class ClimatesetDataHP(torch.utils.data.Dataset):
    """
    Climate dataset with HEALPix projection, similar to DataHP for weather forecasting.
    
    This dataset:
    1. Loads input4mips data (emission scenarios)
    2. Loads CMIP6 data (climate model outputs)
    3. Transforms both to HEALPix grid
    4. Normalizes the data
    5. Returns paired (input, target) samples
    """
    
    # Mapping from gas names to folder names
    GAS_FOLDER_MAPPING = {
        "BC": "BC_sum",
        "CH4": "CH4_sum",
        "SO2": "SO2_sum",
        "CO2": "CO2_sum",
    }
    # Variables that don't have fire type variants
    NO_FIRE_VARS = ["CO2"]
    
    def __init__(self, config: ClimatesetHPConfig):
        # NOTE: should probably call from env().paths.datasets instead
        # han har get cache dir ist'llet
        # statistics path
        self.config = config
        
        # Setup paths
        self.input_dir = Path(config.data_dir) / "inputs" / "input4mips"
        self.target_dir = Path(config.data_dir) / "outputs" / "CMIP6"
        self.cache_dir = Path(config.data_dir) / "cache" / config.cache_name()
        self.stats_path = self.cache_dir / f"{config.statistics_name()}.npy"
        
        # Get year lists
        self.years = config.get_years_list(config.years)
        if config.historical_years is not None:
            self.historical_years = config.get_years_list(config.historical_years)
        else:
            self.historical_years = []
        
        # Load or create data
        # NOTE: Nu laddar den allts[ in datan vid initiering] 'r det bra
        if config.cache and self.cache_dir.exists():
            print(f"Loading cached data from {self.cache_dir}")
            self._load_from_cache()
        else:
            print("Creating dataset from raw files...")
            self._create_dataset()
            print("Dataset created.")
            if config.cache:
                print("Saving dataset to cache...")
                self._save_to_cache()
    
    @staticmethod
    def data_spec(config: ClimatesetHPConfig) -> ClimatesetDataSpec:
        """Get data specification"""
        return ClimatesetDataSpec(
            nside=config.nside,
            n_input_channels=len(config.input_vars),
            n_output_channels=len(config.output_vars),
            seq_len=config.seq_len
        )
    
    def _get_input_paths(self, scenario: str) -> Dict[str, List[str]]:
        """Get file paths for input4mips data"""
        gas_files = {gas: [] for gas in self.config.input_vars}
        
        # Determine which years to use
        # TODO: Add option both historical and scenario data
        if scenario == "historical":
            years = self.historical_years
        else:
            years = self.years
        
        for gas in self.config.input_vars:
            folder_name = self.GAS_FOLDER_MAPPING[gas]
            var_dir = self.input_dir / scenario / folder_name / self.config.spatial_res / self.config.temporal_res
            print(f"var_dir: {var_dir}")
            for year in years:
                year_dir = var_dir / str(year)
                files = glob.glob(str(year_dir / "*.nc"))
                
                # Filter by fire type (except for CO2)
                for f in files:
                    if gas in self.NO_FIRE_VARS or self.config.fire_type in f:
                        gas_files[gas].append(f)
        print(f"Input files found: {{k: len(v) for k, v in gas_files.items()}}")
        return gas_files
    
    def _get_output_paths(self, scenario: str) -> Dict[str, List[str]]:
        """Get file paths for CMIP6 data"""
        var_files = {var: [] for var in self.config.output_vars}
        
        # Determine which years to use
        # TODO: Add option both historical and scenario data
        if scenario == "historical":
            years = self.historical_years
        else:
            years = self.years
        
        for var in self.config.output_vars:
            var_dir = (self.target_dir / self.config.climate_model / 
                      self.config.ensemble / scenario / var / 
                      self.config.spatial_res / self.config.temporal_res)
            print(f"var_dir: {var_dir}")
            for year in years:
                year_dir = var_dir / str(year)
                files = glob.glob(str(year_dir / "*.nc"))
                var_files[var].extend(files)
        
        return var_files
    
    def _interpolate_to_hp(self, data_array: xr.DataArray, 
                          coord_names: Tuple[str, str] = ("lat", "lon")) -> np.ndarray:
        """
        Interpolate xarray data to HEALPix grid.
        
        Args:
            data_array: xarray DataArray with spatial coordinates
            coord_names: Tuple of (latitude_name, longitude_name)
        
        Returns:
            numpy array with shape (..., n_pixels)
        """
        print(f"Data array shape: {data_array.shape}")
        print(f"Data array type: {type(data_array)}")
        print(f"Has chunks: {data_array.chunks if hasattr(data_array, 'chunks') else 'No'}")
        
        print("preparing to interpolate to HEALPix...")
        start = time.time()
        lat_name, lon_name = coord_names
        
        npix = healpix.nside2npix(self.config.nside)
        hlong, hlat = healpix.pix2ang(
            self.config.nside, 
            np.arange(0, npix, 1), 
            lonlat=True, 
            nest=True
        )
        hlong = np.mod(hlong, 360)
        
        xlong = xr.DataArray(hlong, dims="z")
        xlat = xr.DataArray(hlat, dims="z")
        print(f"Preparation took {time.time()-start:.2f}s")
        start = time.time()
        # Interpolate to HEALPix coordinates
        xhp = data_array.interp(
            {lat_name: xlat, lon_name: xlong},
            kwargs={"fill_value": None}
        )
        print(f"Interpolation took {time.time()-start:.2f}s")

        start = time.time()
        hp_image = np.array(xhp.to_numpy(), dtype=np.float32) # NOTE: NO to array here see if it works still
        print(f"Conversion to numpy took {time.time()-start:.2f}s")
        return hp_image
    
    def _load_and_transform_data(self, file_dict: Dict[str, List[str]], 
                                 coord_names: Tuple[str, str] = ("lat", "lon")) -> Dict[str, np.ndarray]:
        """
        Load netCDF files and transform to HEALPix for all variables.
        
        Args:
            file_dict: Dictionary mapping variable names to file lists
            coord_names: Tuple of coordinate names for interpolation
        
        Returns:
            Dictionary mapping variable names to numpy arrays (time, n_pixels)
        """
        hp_data_dict = {}
        
        for var, var_files in file_dict.items():
            print(f"Loading {var}: {len(var_files)} files", flush=True)
            
            t0 = time.time()
            datasets = [xr.open_dataset(f) for f in var_files]
            print(f"  Open files: {time.time()-t0:.2f}s", flush=True)
            
            t0 = time.time()
            ds = xr.concat(datasets, dim="time").sortby("time")
            print(f"  Concat: {time.time()-t0:.2f}s", flush=True)
            
            t0 = time.time()
            arr = ds.to_array().squeeze("variable")
            print(f"  To array: {time.time()-t0:.2f}s", flush=True)
            
            t0 = time.time()
            hp_data = self._interpolate_to_hp(arr, coord_names)
            print(f"  Interpolate to HEALPix: {time.time()-t0:.2f}s", flush=True)
            
            hp_data_dict[var] = hp_data
        
        return hp_data_dict
    
    def _create_dataset(self):
        """Create the full dataset from raw files"""
        all_inputs = []
        all_outputs = []
        
        # Process each scenario
        for scenario in self.config.scenarios:
            print(f"\nProcessing scenario: {scenario}")
            
            # Get file paths
            input_paths = self._get_input_paths(scenario)
            output_paths = self._get_output_paths(scenario)
            
            print(f"Input paths: { {k: len(v) for k, v in input_paths.items()} }")
            print(f"Output paths: { {k: len(v) for k, v in output_paths.items()} }")
            # Load and transform to HEALPix
            print("\nLoading and transforming input data...")
            input_hp = self._load_and_transform_data(input_paths, coord_names=("lat", "lon"))
            print("Loading and transforming output data...")
            output_hp = self._load_and_transform_data(output_paths, coord_names=("y", "x"))
            
            # Stack variables in the order specified in config
            print("\nStacking variables...")
            input_stack = np.stack([input_hp[var] for var in self.config.input_vars], axis=1)
            output_stack = np.stack([output_hp[var] for var in self.config.output_vars], axis=1)
            
            print(f"  Input shape for scenario {scenario}: {input_stack.shape}")
            all_inputs.append(input_stack)
            all_outputs.append(output_stack)
        print("All scenarios processed.")
        # Concatenate all scenarios
        print("\nConcatenating all scenarios...")
        start = time.time()
        self.input_data = np.concatenate(all_inputs, axis=0)  # (n_timesteps, n_input_channels, n_pixels)
        self.output_data = np.concatenate(all_outputs, axis=0)  # (n_timesteps, n_output_channels, n_pixels)
        print(f"Operation took {time.time()-start:.2f}s")
        print(f"\nDataset created:")
        print(f"  Input shape: {self.input_data.shape}")
        print(f"  Output shape: {self.output_data.shape}")
        
        # Compute and apply normalization if needed
        if self.config.normalized:
            start = time.time()
            print("\nNormalizing data...")
            self._compute_and_apply_normalization()
            print("Data normalized.")
            print(f"Operation took {time.time()-start:.2f}s")
    
    def _compute_and_apply_normalization(self):
        """Compute statistics and normalize the data (similar to ERA5 approach)"""
        print("\nComputing normalization statistics...")
        
        # Compute mean and std across time and space, per channel
        input_mean = self.input_data.mean(axis=(0, 2), keepdims=True)  # (1, n_channels, 1)
        input_std = self.input_data.std(axis=(0, 2), keepdims=True)
        
        output_mean = self.output_data.mean(axis=(0, 2), keepdims=True)
        output_std = self.output_data.std(axis=(0, 2), keepdims=True)
        
        # Store statistics
        self.stats = {
            'input_mean': input_mean,
            'input_std': input_std,
            'output_mean': output_mean,
            'output_std': output_std,
        }
        
        # Apply normalization
        self.input_data = (self.input_data - input_mean) / (input_std + 1e-9)
        self.output_data = (self.output_data - output_mean) / (output_std + 1e-9)
        
        print("Normalization applied:")
        for var, mean, std in zip(self.config.input_vars, input_mean[0, :, 0], input_std[0, :, 0]):
            print(f"  {var}: mean={mean:.6e}, std={std:.6e}")
        for var, mean, std in zip(self.config.output_vars, output_mean[0, :, 0], output_std[0, :, 0]):
            print(f"  {var}: mean={mean:.6e}, std={std:.6e}")
    
    def _save_to_cache(self):
        """Save processed data and statistics to cache"""
        print(f"\nSaving to cache: {self.cache_dir}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.save(self.cache_dir / "input_data.npy", self.input_data)
        np.save(self.cache_dir / "output_data.npy", self.output_data)
        
        # Save statistics if normalized
        if self.config.normalized and hasattr(self, 'stats'):
            np.save(self.stats_path, self.stats)
        
        # Save configuration
        config_dict = {
            'nside': self.config.nside,
            'input_vars': self.config.input_vars,
            'output_vars': self.config.output_vars,
            'scenarios': self.config.scenarios,
            'seq_len': self.config.seq_len,
            'normalized': self.config.normalized,
        }
        with open(self.cache_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print("Cache saved successfully")
    
    def _load_from_cache(self):
        # FInns inte i andra
        """Load processed data from cache"""
        self.input_data = np.load(self.cache_dir / "input_data.npy")
        self.output_data = np.load(self.cache_dir / "output_data.npy")
        
        if self.config.normalized and self.stats_path.exists():
            self.stats = np.load(self.stats_path, allow_pickle=True).item()
        
        print(f"Loaded from cache:")
        print(f"  Input shape: {self.input_data.shape}")
        print(f"  Output shape: {self.output_data.shape}")

    def get_statistics_path(self):
        return env().paths.datasets / f"{self.config.statistics_name()}.npy"
    
    def denormalize_output(self, normalized_output: np.ndarray) -> np.ndarray:
        # TODO: Is this needed and used in any script? Is statistics stored elsewhere
        """
        Denormalize output predictions (useful for evaluation).
        
        Args:
            normalized_output: Normalized predictions with shape (..., n_output_channels, n_pixels)
        
        Returns:
            Denormalized predictions
        """
        if not self.config.normalized or not hasattr(self, 'stats'):
            return normalized_output
        
        return (normalized_output * self.stats['output_std'] + 
                self.stats['output_mean'])
    
    def get_meta(self) -> Dict[str, Dict]:
        # Finns i hans kod ocks[]
        # TODO: I have not actually specified units anywhere these are probably not right
        """Get metadata about variables"""
        return {
            'input': {
                'names': self.config.input_vars,
                'units': ['kg/m2/s' if var != 'CO2' else 'ppm' for var in self.config.input_vars],
            },
            'output': {
                'names': self.config.output_vars,
                'units': ['K' if var == 'tas' else 'kg/m2/s' for var in self.config.output_vars],
            }
        }
    
    # def __len__(self):
    #     """
    #     # TODO: Should we use more than one sample
    #     Han har inte
    #     Number of valid samples considering sequence length.
    #     """
    #     return len(self.input_data) - self.config.seq_len + 1

    
    # def __getitem__(self, idx): # NOTE: MUst match batch keys in model forward
    #     """
    #     Get a single sample.
    #     #TODO: We are here getting 12 timesteps, should we get one or not. What should be expected behavior
        
    #     Returns:
    #         dict with keys:
    #             - 'input': input data (seq_len, n_input_channels, n_pixels) or (n_input_channels, n_pixels)
    #             - 'target': target data (seq_len, n_output_channels, n_pixels) or (n_output_channels, n_pixels)
    #             - 'sample_id': index
    #     """

    #     input_seq = self.input_data[idx:idx + self.config.seq_len]
    #     target_seq = self.output_data[idx:idx + self.config.seq_len]
        
    #     # Convert to (seq_len, channels, pixels) if channels_last is False
    #     # or (seq_len, pixels, channels) if channels_last is True
    #     if self.config.channels_last:
    #         input_seq = np.moveaxis(input_seq, 1, -1)
    #         target_seq = np.moveaxis(target_seq, 1, -1)

        
    #     return {
    #         'input': input_seq.astype(np.float32),
    #         'target': target_seq.astype(np.float32),
    #         'sample_id': idx,
    #     }

    def __len__(self):
        """Total number of timesteps available."""
        return len(self.input_data)

    def __getitem__(self, idx):
        """
        Get a single sample (single timestep).
        
        Returns:
            dict with keys:
                - 'input': input data (n_input_channels, n_pixels)
                - 'target': target data (n_output_channels, n_pixels)
                - 'sample_id': index
        """
        input_sample = self.input_data[idx]  # Single timestep
        target_sample = self.output_data[idx]  # Single timestep
        
        # Handle channels_last if needed
        if self.config.channels_last:
            input_sample = np.moveaxis(input_sample, 0, -1)
            target_sample = np.moveaxis(target_sample, 0, -1)
        
        return {
            'input': input_sample.astype(np.float32),
            'target': target_sample.astype(np.float32),
            'sample_id': idx,
        }

def create_climate_dataloaders(
    config: ClimatesetHPConfig,
    train_years: str = "2015-2080",
    val_years: str = "2081-2090",
    test_years: str = "2091-2100",
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Base configuration
        train_years: Year range for training
        val_years: Year range for validation
        test_years: Year range for testing
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create configs for each split
    train_config = copy.deepcopy(config)
    train_config.years = train_years
    
    val_config = copy.deepcopy(config)
    val_config.years = val_years
    
    test_config = copy.deepcopy(config)
    test_config.years = test_years
    
    # Create datasets
    train_dataset = ClimatesetDataHP(train_config)
    val_dataset = ClimatesetDataHP(val_config)
    test_dataset = ClimatesetDataHP(test_config)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    config = ClimatesetHPConfig(
        nside=32,
        data_dir="/proj/heal_pangu/users/x_tagty/climateset",
        climate_model="CAS-ESM2-0",
        ensemble="r3i1p1f1",
        input_vars=["BC", "CH4", "SO2", "CO2"],
        output_vars=["tas", "pr"],
        scenarios=["historical", "ssp585"],
        years="2015-2100",
        historical_years="1850-2014",
        seq_len=12,
        seq_to_seq=True,
        normalized=True,
        cache=True
    )

    print("\nActual threading test:")
    import time
    size = 5000
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)

    t0 = time.time()
    c = np.dot(a, b)
    elapsed = time.time() - t0
    print(f"Matrix multiply ({size}x{size}): {elapsed:.3f}s")
        
    # Create dataset
    dataset = ClimatesetDataHP(config)
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Data spec: {ClimatesetDataHP.data_spec(config)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Input shape: {sample['input'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_climate_dataloaders(
        config,
        batch_size=8
    )
    
    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")