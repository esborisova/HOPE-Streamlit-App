import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pyplot_themes as themes
from wordcloud import WordCloud, STOPWORDS
import string
import pandas as pd
import os
import sys
import spacy


def read_pkl(argument,
             path: str,
             data_prefix: str):
    """Reads pickle file
    
    
    Args:
        argument: The argument passed to the bash script

        path (str): The path to the folder with files 
    
        data_prefix (str): The file name prefix 
    
    
    Returns:
          pd.DataFrame: The pandas dataframe with data
    """

    label = argument.lower()

    filename =  label + data_prefix
    new_path = path + label + '/' + filename

    df = pd.read_pickle(new_path)  

    return df


def plot_wordcloud(wordcloud,
                   argument,
                   path: str,
                   data_prefix: str,
                   save: bool):

    """Plots and saves wordcloud 
    
    
    Args:
        wordcloud: An object of class WordCloud

        argument: An argument passed to the bash script

        path (str): The path to the folder with files 
    
        data_prefix (str): The file name prefix 
        
        save (bool): A boolean for saving the wordcloud
    
    """
    
    plt.figure(figsize=(40, 30))
    plt.axis("off")
    plt.imshow(wordcloud)

    label = argument.lower()
    new_path = path + label + '/'
    filename =  label + data_prefix
    new_path = new_path + filename

    if save:
        plt.savefig(path)






df = read_pkl(str(sys.argv[1]), 'data/', '.pkl')


texts = df["tokens_string"]
texts = texts[texts.notnull()] 
texts = ", ".join(texts)


sp = spacy.load('da_core_news_lg')
file = open("../stops_lemmas.txt","r+")
stop_words = file.read().split()



wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                      background_color='white', colormap="rocket", 
                      collocations=False, stopwords = stop_words).generate(texts)


plot_wordcloud(wordcloud, str(sys.argv[1]), 'data/', '_wordcloud.png', save = True)









