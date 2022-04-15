"""A pipeline for creating WordClouds"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import sys
import spacy
from scripts import read_pkl, plot_wordcloud


df = read_pkl(str(sys.argv[1]), "../data/", ".pkl")

texts = df["tokens_string"]
texts = texts[texts.notnull()]
texts = ", ".join(texts)

sp = spacy.load("da_core_news_lg")
file = open("../stops_lemmas.txt", "r+")
stop_words = file.read().split()

wordcloud = WordCloud(
    width=3000,
    height=2000,
    random_state=1,
    background_color="white",
    colormap="rocket",
    collocations=False,
    stopwords=stop_words,
).generate(texts)


plot_wordcloud(wordcloud, str(sys.argv[1]), "../data/", "_wordcloud.png", save=True)
