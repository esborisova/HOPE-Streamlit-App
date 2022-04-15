"""Bigram frequency calculation and scaling """
import math
from typing import Dict


def bigram_freq(freq: Dict[str, int], G, scale: bool) -> Dict[str, int]:
    """
    Creates a dictionary of words (present in bigrams) and their frequencies.

    Args:
        freq (Dict[str, int]): A dictionary with all words from the data corpus and their frequency values.
        G: A Networkx graph.
        scale (bool): If true, the logarithm of frequency values will be calculated.

    Returns:
        Dict[str, int]: The dictionary containg words from bigrams and their frequencies.
    """

    freq_dict = {}

    for node in G.nodes():
        for word in freq:
            if word[0] == node:
                if scale == True:
                    freq_dict[word[0]] = (math.log2(word[1])) * 3
                else:
                    freq_dict[word[0]] = word[1]
    return freq_dict


def scale(data: Dict[int]) -> Dict[int]:
    """
    Calculates logarithm base 2 and multiplies the result by 3.

    Args:
        data (Dict[int]): A ditctionary with values.

    Returns:
        Dict[int]: The input dictonary with updated values.
    """

    for key, value in data.items():
        data[key] = (math.log2(value)) * 3

    return data
