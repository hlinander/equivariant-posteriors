"""Extract one real validation sample pair (t=0 and t=0+24h) from ERA5/HEALPix.

Run on the cluster from the repo root:

    uv run python experiments/weather/extract_sample_pair.py --date 2019-06-01

Writes a single .npz with the normalized input/target fields plus the dataset
mean/std so the fields can be denormalized offline. Copy it back with e.g.:

    scp berzelius:.../figure_sample_2019-06-01.npz experiments/weather/
"""

import argparse
from datetime import datetime

import numpy as np

from experiments.weather.data import (
    DataHP,
    DataHPConfig,
    ERA5_START_YEAR_TEST,
    deserialize_dataset_statistics,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="2019-06-01",
        help="validation date (YYYY-MM-DD, within the test year)",
    )
    parser.add_argument("--nside", type=int, default=64)
    parser.add_argument("--out", default=None, help="output .npz path")
    args = parser.parse_args()

    config = DataHPConfig(nside=args.nside).validation()
    ds = DataHP(config)

    date = datetime.strptime(args.date, "%Y-%m-%d")
    index = (date - datetime(ERA5_START_YEAR_TEST, 1, 1)).days
    if not 0 <= index < len(ds):
        raise SystemExit(
            f"{args.date} maps to index {index}, outside the validation set "
            f"(0..{len(ds) - 1}, year {ERA5_START_YEAR_TEST})"
        )

    item = ds[index]
    stats = deserialize_dataset_statistics(args.nside).item()

    out = args.out or f"experiments/weather/figure_sample_{args.date}.npz"
    np.savez_compressed(
        out,
        input_surface=item["input_surface"],
        input_upper=item["input_upper"],
        target_surface=item["target_surface"],
        target_upper=item["target_upper"],
        time=item["time"],
        prediction_timedelta_hours=item["prediction_timedelta_hours"],
        normalized=config.normalized,
        mean_surface=stats["mean_surface"],
        std_surface=stats["std_surface"],
        mean_upper=stats["mean_upper"],
        std_upper=stats["std_upper"],
    )
    for key in ["input_surface", "input_upper", "target_surface", "target_upper"]:
        print(f"{key}: {item[key].shape} {item[key].dtype}")
    print(f"time: {item['time']} (+{item['prediction_timedelta_hours']}h target)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
