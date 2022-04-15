"""Setting parameters depending on the label"""
import datetime
import pandas as pd
from typing import List, Dict, Tuple


def read_pkl(label: str, path: str, data_prefix: str) -> pd.DataFrame:
    """
    Reads a pickle file.

    Args:
        label (str): The name of the selectbox (Restriktion, Omicron, etc.).
        path (str): The path to the folder with files.
        data_prefix (str): The file name prefix (restriktion_hash.pkl: Label — restriktion, prefix — _hash.pkl).

    Returns:
        pd.DataFrame: The pandas dataframe with data.
    """

    label = label.lower()

    filename = label + data_prefix
    new_path = path + label + "/" + filename

    df = pd.read_pickle(new_path)

    return df


def smoothing(label: str) -> int:
    """
    Defines the smoothing value depending on the dataset.

    Args:
        label (str): The name of the dataset (from the menu).

    Returns:
        int: The smoothing value.
    """

    smoothing_value = 0

    if (label == "Vaccin") or (label == "Corona"):
        smoothing_value = 5000
    else:
        smoothing_value = 500

    return smoothing_value


def set_range_freq(label: str) -> List[str]:
    """
    Defines the (date) range of X axis for the tweet frequency graph.

    Args:
        label (str): The name of the dataset (from the menu).

    Returns:
        List[str]: The range of X axis.
    """

    xaxis_range = None

    if label == "Omicron":
        xaxis_range = [datetime.datetime(2021, 11, 1), datetime.datetime(2022, 1, 31)]

    elif (label == "Vaccin") or (label == "Corona"):
        xaxis_range = [datetime.datetime(2020, 11, 1), datetime.datetime(2022, 1, 31)]

    else:
        xaxis_range = [datetime.datetime(2020, 11, 1), datetime.datetime(2022, 1, 31)]

    return xaxis_range


def set_lab_vader(label: str) -> Tuple[str, list]:
    """
    Defines the name of the dataframe column with smoothed vader sentiment
    scores and the (date) range of X axis.

    Args:
        label (str): The name of the dataset (from the menu).

    Returns:
        Tuple[str, list]: The label for Y values and the range of X axis.
    """

    y2_name = None
    xaxis_range = None

    if label == "Omicron":
        y2_name = "s500_compound"
        xaxis_range = [datetime.datetime(2021, 11, 1), datetime.datetime(2022, 1, 31)]

    elif (label == "Vaccin") or (label == "Corona"):
        y2_name = "s5000_compound"
        xaxis_range = [datetime.datetime(2020, 12, 15), datetime.datetime(2022, 1, 25)]

    else:
        y2_name = "s500_compound"
        xaxis_range = [datetime.datetime(2020, 12, 15), datetime.datetime(2022, 1, 25)]
    return y2_name, xaxis_range


def set_lab_bert(label: str) -> Tuple[str, list]:
    """
    Defines the name of the dataframe column with smoothed bert sentiment
    scores and the (date) range of X axis.

    Args:
        label (str): The name of the dataset (from the menu).

    Returns:
        Tuple[str, list]: The label for Y values and the range of X axis.
    """

    y2_name = None
    xaxis_range = None

    if label == "Omicron":
        y2_name = "s500_polarity_score_z"
        xaxis_range = [datetime.datetime(2021, 11, 1), datetime.datetime(2022, 1, 31)]

    elif (label == "Vaccin") or (label == "Corona"):
        y2_name = "s5000_polarity_score_z"
        xaxis_range = [datetime.datetime(2020, 12, 15), datetime.datetime(2022, 1, 25)]

    else:
        y2_name = "s500_polarity_score_z"
        xaxis_range = [datetime.datetime(2020, 12, 15), datetime.datetime(2022, 1, 25)]

    return y2_name, xaxis_range


def choose_keywords(label: str, keywords_dict: Dict[str]) -> List[str]:
    """
    Returns the relevant list of keywords.

    Args:
        label (str): The name of the dataset (from the menu).
        keywords_dict (Dict[str]): The dictionary with keywords per dataset.

    Returns:
        List[str]: The list of keywords.
    """
    for i in keywords_dict:
        if i == label:
            return keywords_dict[i]
