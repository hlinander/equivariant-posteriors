import experiments.weather.cdsmontly as cdstest
import numpy as np
import calendar
import sys, subprocess




# Downloading for the year 2019
year = 2019
year_str = str(year)
for month in np.arange(1, 13, 1):
    month_str = f"{month:02d}"

    subprocess.run("python run_slurm.py --cpu experiments/weather/download_lite_month.py " + year_str + " " + month_str + " 2",
                    shell=True,
                    executable="/bin/bash")

    
    