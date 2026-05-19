import experiments.weather.cdsmontly as cdstest
import numpy as np
import calendar
import sys, subprocess



# Downloading for the year 2019
year = 2012
year_str = str(year)


if (len(sys.argv) == 2 and sys.argv[1] == "--debug"):
    subprocess.run("uv run experiments/weather/download_lite_month.py " + year_str + " 1" + " 2",
                    shell=True,
                    executable="/bin/bash")
    
for month in np.arange(1, 13, 1):
    month_str = f"{month:02d}"

    subprocess.run("uv run run_slurm.py --cpu experiments/weather/download_lite_month.py " + year_str + " " + month_str + " 2",
                    shell=True,
                    executable="/bin/bash")

    
    