"""Functions for pkl files, computing CI, plotting WordClouds """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys


def read_pkl(argument, path: str, data_prefix: str):
    """
    Reads a pickle file.

    Args:
        argument: An argument passed to the bash script.
        path (str): The path to the folder with files
        data_prefix (str): The file name prefix.

    Returns:
        pd.DataFrame: The pandas dataframe with data.
    """

    label = argument.lower()

    filename = label + data_prefix
    new_path = path + label + "/" + filename

    df = pd.read_pickle(new_path)

    return df


def save_pkl(dataset: pd.DataFrame, argument, path: str, data_prefix: str):
    """
    Saves data to a pickle file.

    Args:
        dataset (pd.DataFrame): The dataframe to save.
        argument: The argument passed to the bash script.
        path (str): The path to the main folder.
        data_prefix (str): The file name prefix.
    """

    label = argument.lower()
    new_path = path + label + "/"
    filename = label + data_prefix
    new_path = new_path + filename

    dataset.to_pickle(new_path)


def compute_ci(
    data: pd.DataFrame, array: str, upper_bound: str, lower_bound: str
) -> pd.DataFrame:
    """
    Computes 95% confidence interval (CI).

    Args:
        data (pd.DataFrame): The dataframe with the dataset.
        array (str): The dataframe column with arrays of values.
        upper_bound (str): The name of the output dataframe column with CI upper bound values.
        lower_bound (str): The name of the output dataframe column with CI lower bound values.

    Returns:
        pd.DataFrame: The input dataframe with two new columns (upper and lower bound values).
    """

    data[upper_bound] = 0.0
    data[lower_bound] = 0.0

    for i in range(len(data)):

        CI = sns.utils.ci(sns.algorithms.bootstrap(data[array][i]))

        data[upper_bound][i] = CI[0]
        data[lower_bound][i] = CI[1]

    return data


def plot_wordcloud(wordcloud, argument, path: str, data_prefix: str, save: bool):
    """
    Plots and saves a wordcloud.

    Args:
        wordcloud: An object of class WordCloud.
        argument: An argument passed to the bash script.
        path (str): The path to the folder with files.
        data_prefix (str): The file name prefix.
        save (bool): If True, saves a wordcloud.
    """

    plt.figure(figsize=(40, 30))
    plt.axis("off")
    plt.imshow(wordcloud)

    label = argument.lower()
    new_path = path + label + "/"
    filename = label + data_prefix
    new_path = new_path + filename

    if save:
        plt.savefig(path)


def set_lab_freq(label: str):

    column_name = None

    if (label == "vaccin") or (label == "corona"):
        column_name = "s5000_nr_of_tweets"

    else:
        column_name = "s500_nr_of_tweets"

    return column_name
