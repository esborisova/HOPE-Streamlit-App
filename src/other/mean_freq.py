"""Pipeline for collecting mean frequency values"""
import pandas as pd
import sys
from scripts import read_pkl, save_pkl, set_lab_freq


df = read_pkl(str(sys.argv[1]), "../data/", ".pkl")

name = set_lab_freq(str(sys.argv[1]))

mean = df.groupby("date")[name].mean()
mean = mean.reset_index()
mean = mean.rename(columns={name: "mean"})

merged = pd.merge(mean, df, on="date")

merged = merged.filter(["date", "nr_of_tweets", "mean"], axis=1)

save_pkl(merged, str(sys.argv[1]), "../data/", "_tweet_freq.pkl")
