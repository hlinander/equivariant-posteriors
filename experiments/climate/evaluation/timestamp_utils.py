"""
Utilities for mapping between raw dataset array offsets (sample_id) and
human-readable timestamps (scenario, year, month).

Works for both HP and nohp datasets — both use the same concatenation scheme:
  raw_array = concat over scenarios in config.scenarios order, then time (monthly).

Usage in notebooks
------------------
    from experiments.climate.evaluation.timestamp_utils import (
        sample_id_to_timestamp,
        find_sample_for_timestamp,
        months_abbr,
    )
"""


MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def sample_id_to_timestamp(sample_id, data_config):
    """
    Map a raw array offset (sample_id from the batch dict) to (scenario, year, month).

    Works for both ClimatesetHPConfig and ClimatesetConfig.

    Returns
    -------
    scenario : str   e.g. "ssp245"
    year     : int   e.g. 2040
    month    : int   1-indexed (1=Jan … 12=Dec)
    """
    years = data_config.get_years_list(data_config.years)
    scenarios = list(data_config.scenarios)
    months_per_scenario = len(years) * 12

    scenario_idx = int(sample_id) // months_per_scenario
    month_offset = int(sample_id) % months_per_scenario

    scenario = scenarios[scenario_idx]
    year = years[0] + month_offset // 12
    month = month_offset % 12 + 1
    return scenario, year, month


def find_sample_for_timestamp(year, month, data_config):
    """
    Given a (year, month) and the test data config, return how to index into
    the predictions array returned by get_sample_predictions / get_sample_predictions_nohp.

    Returns
    -------
    dataloader_idx : int
        Index into the N dimension of the returned predictions array.
        For seq_len=1 this equals the raw_offset.
        For seq_len=12 (non-overlapping test chunks) this is raw_offset // seq_len.
    seq_pos : int
        Position within the temporal sequence to extract.
        Always 0 for seq_len=1.
        0-11 for seq_len=12 (0=Jan of that year … 11=Dec).
    raw_offset : int
        Absolute timestep index in the raw concatenated array (= sample_id).
    """
    years = data_config.get_years_list(data_config.years)
    start_year = years[0]
    raw_offset = (year - start_year) * 12 + (month - 1)
    seq_len = getattr(data_config, "seq_len", 1)
    dataloader_idx = raw_offset // seq_len
    seq_pos = raw_offset % seq_len
    return dataloader_idx, seq_pos, raw_offset


def format_timestamp(scenario, year, month):
    """Human-readable string, e.g. 'ssp245 · Apr 2040'."""
    return f"{scenario} · {MONTH_NAMES[month - 1]} {year}"
