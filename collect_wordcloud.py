import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pyplot_themes as themes
from wordcloud import WordCloud, STOPWORDS
import string
import pandas as pd
import os
import sys
import spacy


def read_file(argument,
              folder: str,
              data_prefix: str):

    label = argument.lower()
    df = pd.read_pickle('data/' + label + folder + label + data_prefix)  

    return df



def plot_cloud(wordcloud,
               argument,
               save=False):
    
    plt.figure(figsize=(40, 30))
    plt.axis("off")
    plt.imshow(wordcloud)
    label = argument.lower()
    path = 'data/' + label + '_streamlit/' + label + '_wordcloud.png'

    if save:
        plt.savefig(path)



df = read_file(str(sys.argv[1]), '_streamlit/', '.pkl')


texts = df["tokens_string"]
texts = texts[texts.notnull()] 
texts = ", ".join(texts)


sp = spacy.load('da_core_news_lg')
file = open("stops_lemmas.txt","r+")
stop_words = file.read().split()


wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, 
                      background_color='white', colormap="rocket", 
                      collocations=False, stopwords = stop_words).generate(texts)

plot_cloud(wordcloud, str(sys.argv[1]), save=True)









