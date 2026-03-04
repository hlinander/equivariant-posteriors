import experiments.weather.cdsmontly as cdstest
import numpy as np
import calendar

# Downloading for the year 2019
year = 2019
year_str = str(year)
for month in np.arange(1, 13, 1):
    month_str = f"{month:02d}"
    num_days = calendar.monthrange(year, month)[1]
    for day in np.arange(1, num_days + 1, 1):
        for hour in np.arange(0, 24, 2):
            try:
                e5sc = cdstest.ERA5SampleConfig(
                    year=year_str, month=month_str, day=day, time=f"{hour:02d}:00:00"
                )
                e5s = cdstest.get_era5_sample(e5sc)

                print(f"Downloaded {e5sc.ident()}")
                del e5s
                
            except Exception as e:
                print(f"[Fail] {e}")
