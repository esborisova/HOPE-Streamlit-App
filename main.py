from tkinter import font
from _plotly_utils.utils import find_closest_string
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import datetime
import networkx as nx
from icecream import ic
from nltk.util import bigrams 
import os
from bokeh.palettes import Spectral4
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Range1d, TapTool,
                          Range1d, ColumnDataSource, LabelSet)
from bokeh.plotting import from_networkx

from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap

from functions import *



#################################
# Set up the upper navigate bar #
#################################

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
  <a class="navbar-brand" href="https://hope-project.dk/#/" target="_blank">HOPE Project</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href= "https://github.com/stinenyhus/HOPE-keyword-query-Twitter" target="_blank">GitHub</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html = True)


############################################
# Hide the menu button (on the right side) #
############################################

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True) 


############################################################################
# Condense the layout (fit on the screen to reduce the amount of scrolling)#
############################################################################

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


def main():

    menu = st.sidebar.selectbox('MENU', ['Restriktion', 'Genåb'])
    navigator = st.sidebar.radio('NAVIGATE TO', ['Tweet Frequency', 'Sentiment', 'Hashtags', 'Word Frequency', 'Bigrams', 'WordCloud'])
    

    if navigator == 'Tweet Frequency':
        st.title('Tweet Frequency Over Time') 

        df = read_file(label = menu, 
                       folder = '_streamlit/',
                       data_prefix = '.pkl') 
        

        fig = plot_tweet_freq(data = df,
                              x_column = 'date',
                              y1_column = 'nr_of_tweets',
                              y2_column = 's500_nr_of_tweets',
                              line1_name = 'Number of tweets',
                              line2_name = 'Smoothed values',
                              title = 'Frequency of Mentions')
                                  
        st.plotly_chart(fig, use_container_width = True)

    elif navigator == 'Sentiment':
           st.title('Sentiment Over Time')
           st.write("""
           - Centered sentiment score
           """)

           st.write("""
           - Z polarity score
           """)


    elif navigator == 'Hashtags':
          st.title('Hashtags Frequency') 

          df = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_hash.pkl') 

          df = df.sort_values('nr_of_hashtags', ascending = True)

          fig = plot_bar_freq(data = df,
                               x_column = 'nr_of_hashtags',
                               y_column = 'hashtag',
                               title_text = 'Most Frequent Hashtags',
                               colorscale = 'inferno')

          st.plotly_chart(fig, use_container_width = True)


    elif navigator == 'Word Frequency':
        st.title('Word Frequency')

        df = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_w_freq.pkl') 

        df= df.sort_values('Frequency', ascending = True)

        fig = plot_bar_freq(data = df,
                            x_column = 'Frequency',
                            y_column = 'word',
                            title_text = 'Most Frequent Words',
                            colorscale = 'brwnyl')

        st.plotly_chart(fig, use_container_width = True)
     

    elif navigator == 'Bigrams':
         st.title('Bigrams')

         df1 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '.pkl') 

         w_freq = word_freq(df1)

         df2 = read_file(label = menu, 
                         folder = '_streamlit/',
                         data_prefix = '_bigrams.pkl') 


         plot = plot_bigrams(data = df2 ,
                             w_freq = w_freq,
                             title = 'Bigrams')

         st.bokeh_chart(plot, use_container_width=True)

    elif navigator == 'WordCloud':
         st.title('WordCloud')

    else:
             raise ValueError('Invalid input data!')
                

if __name__  == '__main__':
    main()   