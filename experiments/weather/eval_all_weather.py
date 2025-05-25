import subprocess
import os

# " BATCHSIZE=1 LEADTIME=2 GPU=A40:1 ./srun_single.sh ./run.sh experiments/weather/evaluate.py experiments/weather/persisted_configs/train_nside64.py 10"

models = [
    # "experiments/weather/persisted_configs/train_nside64.py",
    # "experiments/weather/persisted_configs/train_nside64_mask.py",
    # "experiments/weather/persisted_configs/train_nside64_single_relpos.py",
    # "experiments/weather/persisted_configs/train_pangu_nside64.py",
    "experiments/weather/persisted_configs/train_pangu_nside64_adapted.py",
    # "experiments/weather/persisted_configs/train_nside64_single_relpos_fixed_patch_size.py",
    # "experiments/weather/persisted_configs/train_nside64_single_relpos_pad.py",
    # "experiments/weather/persisted_configs/train_nside64_single_relpos_pad_and_patch.py"
]

env = os.environ.copy()

# for epoch in [100, 200, 300]:  # range(100, 200, 10):
#     for lead_time in range(10):
#         for model in models:
#             env["BATCHSIZE"] = "1"
#             env["LEADTIME"] = f"{lead_time}"
#             env["GPU"] = "A40:1"
#             cmd = [
#                 "./srun_single.sh",
#                 "./run.sh",
#                 "experiments/weather/evaluate.py",
#                 model,
#                 f"{epoch}",
#             ]
#             subprocess.run(cmd, env=env)
#             # print(cmd)
#             # input()

for epoch in [250]:  # range(100, 200, 10):
    for lead_time in range(10):
        for model in models:
            env["BATCHSIZE"] = "1"
            env["LEADTIME"] = f"{lead_time}"
            env["GPU"] = "A40:1"
            cmd = [
                "./srun_single.sh",
                "./run.sh",
                "experiments/weather/evaluate.py",
                model,
                f"{epoch}",
            ]
            subprocess.run(cmd, env=env)
