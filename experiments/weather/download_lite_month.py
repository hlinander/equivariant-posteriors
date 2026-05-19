import experiments.weather.cdsmontly as cdstest
import numpy as np
import sys
import calendar


def download_monthly(year: int, month: int, delta_t: int):
    """
    Download every day for a given year and month, with a specified hour interval (delta_t).
    """
    year_str = str(year)
    month_str = f"{month:02d}"
    for hour in np.arange(0, 24, delta_t):
        try:
            # Assuming that it will download the sample for every day of the month at the specified hour
            e5sc = cdstest.ERA5SampleConfig(
                year=year_str, month=month_str, day="1", time=f"{hour:02d}:00:00"
            )
            e5s = cdstest.get_era5_sample(e5sc)

            print(f"Downloaded {e5sc.ident()}")
            del e5s

        except Exception as e:
            print(f"[Fail] {e}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(sys.argv)
        print("Usage: python download_lite_month.py <year> <month> <delta_t>")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    delta_t = int(sys.argv[3])

    download_monthly(year, month, delta_t)