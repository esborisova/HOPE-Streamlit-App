import pandas as pd 
import numpy as np
import seaborn as sns
import os
import sys


def compute_ci(data: pd.DataFrame,
               array: str,
               upper_bound: str,
               lower_bound: str):

    data[upper_bound] = 0.0
    data[lower_bound] = 0.0

    for i in range(len(data)):

        CI = sns.utils.ci(sns.algorithms.bootstrap(data[array][i])) 
        
        data[upper_bound][i] = CI[0]
        data[lower_bound][i] = CI[1]
        
    return data



def read_file(argument,
              folder: str,
              data_prefix: str):

    label = argument.lower()
    df = pd.read_pickle('data/' + label + folder + label + data_prefix)  

    return df


def create_pkl(dataset: pd.DataFrame, 
               argument, 
               folder: str,
               data_prefix: str):

    label = argument.lower()

    dataset.to_pickle('data/' + label + folder + label + data_prefix)  

  

df = read_file(str(sys.argv[1]), '_streamlit/', '.pkl')

# Create df with dates and sentiment scores
df0 = df.filter(['date', 'centered_compound', 'polarity_score_z'], axis=1)

# Group compound sentiment scores by date. Make arrays from the grouped values
df_compound = df0.groupby(['date'])['centered_compound'].apply(list).apply(lambda x: np.asarray(x))
df_compound = df_compound.reset_index()
df_compound = df_compound.rename(columns = {'centered_compound':'compound_array'})


# Calculate mean for arrays of compound scores
mean_compound = df0.groupby("date")['centered_compound'].mean()
mean_compound  = mean_compound.reset_index()
mean_compound = mean_compound.rename(columns = {'centered_compound':'compound_mean'})


# Calculate upper and lower bounds for 95% confidence interval for compound score

df_compound = compute_ci(df_compound, 
                         array = 'compound_array',
                         upper_bound = 'compound_upper',
                         lower_bound = 'compound_lower')


# Merge mean and arrays for compund scores into a single df
merged_compound = pd.merge(df_compound, mean_compound, on = 'date')


# Group z sentiment scores by date. Make arrays from the grouped values
df_z = df0.groupby(['date'])['polarity_score_z'].apply(list).apply(lambda x: np.asarray(x))
df_z = df_z.reset_index()
df_z = df_z.rename(columns = {'polarity_score_z':'z_array'})


# Calculate mean for arrays of compound scores
mean_z = df0.groupby("date")['polarity_score_z'].mean()
mean_z  = mean_z.reset_index()
mean_z = mean_z.rename(columns = {'polarity_score_z':'z_mean'})


# Calculate upper and lower bounds for 95% confidence interval for z score
df_z = compute_ci(df_z, 
                  array = 'z_array',
                  upper_bound = 'z_upper',
                  lower_bound = 'z_lower')



# Merge mean and arrays for z scores into a single df
merged_z= pd.merge(df_z, mean_z, on = 'date')

# Create df with all values for z and compound scores 
df_merged = pd.merge(merged_compound, merged_z, on = 'date')

# Save df to pickle
create_pkl(df_merged, str(sys.argv[1]), '_streamlit/', '_sentiment.pkl')