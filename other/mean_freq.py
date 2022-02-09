from fileinput import filename
import pandas as pd 
import numpy as np
import seaborn as sns
import os
import sys



def read_pkl(argument,
              path: str,
              data_prefix: str):


    label = argument.lower()

    filename =  label + data_prefix
    new_path = path + label + '/' + filename

    df = pd.read_pickle(new_path)  

    return df



def save_pkl(dataset: pd.DataFrame, 
             argument, 
             path: str,
             data_prefix: str):

    label = argument.lower()
    new_path = path + label + '/'
    filename =  label + data_prefix
    new_path = new_path + filename

    dataset.to_pickle(new_path)  



def set_lab_freq(label: str):

    column_name = None

    if (label == 'vaccin') or (label == 'corona'):
        column_name= 's5000_nr_of_tweets'
        
    else:
        column_name = 's500_nr_of_tweets'

    return column_name



df = read_pkl(str(sys.argv[1]), '../data/', '.pkl')

name = set_lab_freq(str(sys.argv[1]))

mean = df.groupby('date')[name].mean()
mean = mean.reset_index()
mean = mean.rename(columns = {name:'mean'})

merged  = pd.merge(mean, df, on = 'date')

merged = merged.filter(['date', 'nr_of_tweets', 'mean'], axis=1)

save_pkl(merged, str(sys.argv[1]), '../data/', '_tweet_freq.pkl')
