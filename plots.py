# coding: utf-8
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("webagg")
df = pd.read_pickle("database.df.pickle")
groups = df.groupby(["train_config.model.name", "train_config.model.config.embed_d"])

for name, group in groups:
    plt.plot(group["epoch"], group["accuracy"], label=name)

plt.legend()
plt.show()
