"""A pipeline for collecting mean"""
import pandas as pd
import numpy as np
import sys
from scripts import read_pkl, save_pkl, compute_ci


df = read_pkl(str(sys.argv[1]), "../data/", ".pkl")

# Create df with dates and sentiment scores
df0 = df.filter(["date", "centered_compound", "polarity_score_z"], axis=1)

# Group compound sentiment scores by date. Make arrays from the grouped values
df_compound = (
    df0.groupby(["date"])["centered_compound"]
    .apply(list)
    .apply(lambda x: np.asarray(x))
)
df_compound = df_compound.reset_index()
df_compound = df_compound.rename(columns={"centered_compound": "compound_array"})


# Calculate mean for arrays of compound scores
mean_compound = df0.groupby("date")["centered_compound"].mean()
mean_compound = mean_compound.reset_index()
mean_compound = mean_compound.rename(columns={"centered_compound": "compound_mean"})


# Calculate upper and lower bounds for 95% confidence interval for compound score
df_compound = compute_ci(
    df_compound,
    array="compound_array",
    upper_bound="compound_upper",
    lower_bound="compound_lower",
)

# Merge mean and arrays for compund scores into a single df
merged_compound = pd.merge(df_compound, mean_compound, on="date")

# Group z sentiment scores by date. Make arrays from the grouped values
df_z = (
    df0.groupby(["date"])["polarity_score_z"].apply(list).apply(lambda x: np.asarray(x))
)
df_z = df_z.reset_index()
df_z = df_z.rename(columns={"polarity_score_z": "z_array"})

# Calculate mean for arrays of compound scores
mean_z = df0.groupby("date")["polarity_score_z"].mean()
mean_z = mean_z.reset_index()
mean_z = mean_z.rename(columns={"polarity_score_z": "z_mean"})

# Calculate upper and lower bounds for 95% confidence interval for z score
df_z = compute_ci(df_z, array="z_array", upper_bound="z_upper", lower_bound="z_lower")

# Merge mean and arrays for z scores into a single df
merged_z = pd.merge(df_z, mean_z, on="date")

# Create df with all values for z and compound scores
df_merged = pd.merge(merged_compound, merged_z, on="date")
merged_final = pd.merge(df, df_merged, on="date")
merged_final = merged_final.drop_duplicates(subset=["date"])

# Save df to pickle
save_pkl(merged_final, str(sys.argv[1]), "../data/", "_sentiment.pkl")
