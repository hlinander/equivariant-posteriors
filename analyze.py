import pandas
import glob
import matplotlib.pyplot as plt

dfs = [pandas.read_pickle(path) for path in glob.glob("checkpoints/*.df.pickle")]
df = pandas.concat(dfs)
df.to_pickle("database.df.pickle")
